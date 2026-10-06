// Direct first-answer Bonsai decisions; built against the exact local llama API.
#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
static int observed_offloaded_layers = -1, observed_total_layers = -1;
static void native_log(ggml_log_level, const char * text, void *) {
    if (const char * found = std::strstr(text, "offloaded ")) {
        int offloaded = -1, total = -1;
        if (std::sscanf(found, "offloaded %d/%d layers to GPU", &offloaded, &total) == 2) {
            observed_offloaded_layers = offloaded; observed_total_layers = total;
        }
    }
    std::fputs(text, stderr);
}
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Failure : std::runtime_error {
    explicit Failure(const std::string & kind) : std::runtime_error(kind) {}
};
struct Options {
    std::string model, input, output, probe;
    int ctx = 4096, batch = 2048, ubatch = 512, threads = 8, gpu_layers = 99, warmup = 2;
    bool validate = false, help = false;
};
static int integer(const std::string & value, int minimum) {
    try {
        size_t used = 0;
        long parsed = std::stol(value, &used);
        if (used != value.size() || parsed < minimum || parsed > 1000000) throw Failure("invalid_arguments");
        return int(parsed);
    } catch (...) { throw Failure("invalid_arguments"); }
}
static Options parse_options(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") { o.help = true; continue; }
        if (flag == "--validate-only") { o.validate = true; continue; }
        if (i + 1 >= argc) throw Failure("invalid_arguments");
        const std::string value = argv[++i];
        if (flag == "--model") o.model = value;
        else if (flag == "--input") o.input = value;
        else if (flag == "--output") o.output = value;
        else if (flag == "--sampler-probe") o.probe = value;
        else if (flag == "--ctx-size") o.ctx = integer(value, 1);
        else if (flag == "--batch-size") o.batch = integer(value, 1);
        else if (flag == "--ubatch-size") o.ubatch = integer(value, 1);
        else if (flag == "--threads") o.threads = integer(value, 1);
        else if (flag == "--gpu-layers") o.gpu_layers = integer(value, -1);
        else if (flag == "--warmup") o.warmup = integer(value, 0);
        else throw Failure("invalid_arguments");
    }
    return o;
}
static bool nonempty_string(const json & value) {
    return value.is_string() && !value.get_ref<const std::string &>().empty();
}
static std::vector<json> read_jobs(const std::string & path) {
    try {
        std::ifstream stream(path);
        if (!stream) throw Failure("invalid_jobs");
        std::vector<json> jobs;
        std::set<std::string> ids;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            if (line.size() > 32 * 1024 * 1024) throw Failure("invalid_jobs");
            auto job = json::parse(line);
            if (!job.is_object() || !job.contains("task_id") || !nonempty_string(job["task_id"]) ||
                !job.contains("prompt") || !nonempty_string(job["prompt"]) ||
                !job.contains("labels") || !job["labels"].is_object() ||
                job["labels"].empty() || job["labels"].size() > 35) throw Failure("invalid_jobs");
            if (!ids.insert(job["task_id"].get<std::string>()).second) throw Failure("invalid_jobs");
            std::set<std::string> marks;
            for (auto it = job["labels"].begin(); it != job["labels"].end(); ++it) {
                if (it.key().empty() || !nonempty_string(it.value()) ||
                    !marks.insert(it.value().get<std::string>()).second) throw Failure("invalid_jobs");
            }
            jobs.push_back(std::move(job));
            if (jobs.size() > 10000) throw Failure("invalid_jobs");
        }
        if (jobs.empty()) throw Failure("invalid_jobs");
        return jobs;
    } catch (...) { throw Failure("invalid_jobs"); }
}

// Use llama's actual float sampler, not a replacement double softmax. No model
// evaluation occurs here; this function is covered by synthetic-logit probes.
static json distribution(const std::vector<float> & logits, const json & ids,
                         const std::vector<llama_token> & suppress = {}) {
    json result = {{"ok", false}, {"error_kind", "input"}};
    if (logits.empty() || !ids.is_object() || ids.empty() || ids.size() > 35) return result;
    for (float logit : logits) if (!std::isfinite(logit)) return result;
    std::set<int> unique;
    std::vector<llama_logit_bias> bias;
    for (auto it = ids.begin(); it != ids.end(); ++it) {
        if (!it.value().is_number_integer()) return result;
        int64_t id = it.value().get<int64_t>();
        if (id < 0 || id >= int64_t(logits.size()) || !unique.insert(int(id)).second) return result;
        bias.push_back({llama_token(id), 50.0f});
    }
    for (llama_token id : suppress) {
        if (id < 0 || id >= int64_t(logits.size())) return result;
        bias.push_back({id, -std::numeric_limits<float>::infinity()});
    }
    if (std::set<llama_token>(suppress.begin(), suppress.end()).size() == logits.size()) {
        result["error_kind"] = "distribution"; return result;
    }
    std::vector<llama_token_data> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) candidates.push_back({llama_token(i), logits[i], 0.0f});
    llama_token_data_array array = {candidates.data(), candidates.size(), -1, false};
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_logit_bias(int(logits.size()), int(bias.size()), bias.data()));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(1.0f));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(1));
    llama_sampler_apply(sampler.get(), &array);
    std::sort(array.data, array.data + array.size,
              [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
    const size_t top_n = std::min<size_t>(40, std::max<size_t>(20, ids.size() * 2));
    json reported = json::object(), missing = json::array();
    for (size_t i = 0; i < std::min(top_n, array.size); ++i) {
        if (!std::isfinite(array.data[i].p) || array.data[i].p < 0 || array.data[i].p > 1) {
            result["error_kind"] = "distribution"; return result;
        }
        if (array.data[i].p == 0.0f) break; // Same server zero-probability omission.
        reported[std::to_string(array.data[i].id)] = array.data[i].p;
    }
    double mass = 0;
    for (auto it = ids.begin(); it != ids.end(); ++it) {
        auto key = std::to_string(it.value().get<int>());
        if (!reported.contains(key)) missing.push_back(it.key());
        else mass += reported[key].get<double>();
    }
    result["candidate_ids"] = ids;
    result["missing_options"] = missing;
    result["top_n"] = top_n;
    result["candidate_mass"] = mass;
    if (!missing.empty() || !std::isfinite(mass) || mass <= 0) {
        result["error_kind"] = "distribution"; return result;
    }
    json probabilities = json::object();
    for (auto it = ids.begin(); it != ids.end(); ++it) {
        const double p = reported[std::to_string(it.value().get<int>())].get<double>() / mass;
        if (!std::isfinite(p) || p < 0 || p > 1) {
            result["error_kind"] = "distribution"; return result;
        }
        probabilities[it.key()] = p;
    }
    result["ok"] = true; result["error_kind"] = nullptr; result["probs"] = probabilities;
    return result;
}
static json probe(const std::string & path) {
    try {
        std::ifstream stream(path); auto input = json::parse(stream);
        auto logits = input.at("logits").get<std::vector<float>>();
        auto suppress = input.value("suppress_ids", std::vector<llama_token>{});
        return distribution(logits, input.at("candidate_ids"), suppress);
    } catch (...) { return {{"ok", false}, {"error_kind", "input"}}; }
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, add_special, true);
    if (n == std::numeric_limits<int32_t>::min()) throw Failure("tokenization");
    std::vector<llama_token> tokens(size_t(n < 0 ? -n : n));
    const int actual = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), add_special, true);
    if (actual < 0) throw Failure("tokenization");
    tokens.resize(size_t(actual));
    return tokens;
}
static llama_token mark_id(const llama_vocab * vocab, const std::string & mark) {
    std::vector<std::string> variants;
    if (mark[0] == ' ') {
        auto pos = mark.find_first_not_of(" \t\r\n");
        variants = {mark, pos == std::string::npos ? "" : mark.substr(pos)};
    } else variants = {mark, " " + mark};
    for (const auto & variant : variants) {
        auto ids = tokenize(vocab, variant, false);
        if (ids.size() == 1) return ids[0];
    }
    throw Failure("tokenization");
}
static json evaluate(llama_context * ctx, const llama_vocab * vocab, const json & job, const Options & o) {
    const auto local_start = Clock::now();
    auto start = Clock::now();
    auto tokens = tokenize(vocab, job["prompt"].get<std::string>(), true);
    json candidate_ids = json::object();
    std::set<llama_token> unique;
    for (auto it = job["labels"].begin(); it != job["labels"].end(); ++it) {
        const auto id = mark_id(vocab, it.value().get<std::string>());
        if (!unique.insert(id).second) throw Failure("tokenization");
        candidate_ids[it.key()] = id;
    }
    const double tokenize_ms = elapsed(start);
    if (tokens.empty() || tokens.size() > size_t(o.ctx)) throw Failure("context_limit");
    start = Clock::now();
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
    llama_perf_context_reset(ctx);
    const double reset_ms = elapsed(start);
    llama_batch batch = llama_batch_init(o.batch, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch); throw Failure("batch_allocation");
    }
    double prefill_ms = 0;
    for (size_t offset = 0; offset < tokens.size(); offset += size_t(o.batch)) {
        batch.n_tokens = int(std::min<size_t>(o.batch, tokens.size() - offset));
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[offset + size_t(i)]; batch.pos[i] = int(offset) + i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = offset + size_t(i) + 1 == tokens.size();
        }
        llama_synchronize(ctx);
        start = Clock::now();
        const int status = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        prefill_ms += elapsed(start);
        if (status != 0) { llama_batch_free(batch); throw Failure("decode"); }
    }
    llama_batch_free(batch);
    start = Clock::now();
    const float * raw = llama_get_logits_ith(ctx, -1);
    if (!raw) throw Failure("logits");
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> logits(raw, raw + n_vocab);
    int n_suppress = 0;
    const auto * suppress_ptr = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    std::vector<llama_token> suppress;
    if (n_suppress > 0) suppress.assign(suppress_ptr, suppress_ptr + n_suppress);
    json result = distribution(logits, candidate_ids, suppress);
    const double sample_ms = elapsed(start);
    const auto perf = llama_perf_context(ctx);
    result.update({{"type", "result"}, {"task_id", job["task_id"]}, {"candidate_ids", candidate_ids},
        {"prompt_tokens", tokens.size()}, {"tokenize_ms", tokenize_ms}, {"reset_ms", reset_ms},
        {"prefill_ms", prefill_ms}, {"sample_ms", sample_ms}, {"decision_ms", prefill_ms + sample_ms},
        {"local_total_ms", tokenize_ms + reset_ms + prefill_ms + sample_ms},
        {"local_wall_ms", elapsed(local_start)}, {"perf_prompt_ms", perf.t_p_eval_ms},
        {"perf_eval_ms", perf.t_eval_ms}, {"perf_prompt_tokens", perf.n_p_eval}, {"perf_eval_tokens", perf.n_eval}});
    return result;
}
static void write_row(FILE * stream, const json & row) {
    const auto text = row.dump() + "\n";
    if (std::fwrite(text.data(), 1, text.size(), stream) != text.size() || std::fflush(stream) != 0 ||
        fsync(fileno(stream)) != 0) throw Failure("output_io");
}
int main(int argc, char ** argv) {
    std::unique_ptr<FILE, decltype(&std::fclose)> output(nullptr, std::fclose);
    std::string current_task;
    try {
        const Options o = parse_options(argc, argv);
        if (o.help) {
            std::cout << "bonsai-native --model PATH --input JOBS.jsonl --output NEW.jsonl "
                "[--ctx-size 4096 --batch-size 2048 --ubatch-size 512 --threads 8 --gpu-layers 99 --warmup 2]\n"
                "Offline tests: --validate-only --input JOBS.jsonl; --sampler-probe INPUT.json\n";
            return 0;
        }
        if (!o.probe.empty()) { std::cout << probe(o.probe).dump() << '\n'; return 0; }
        if (o.input.empty()) throw Failure("invalid_arguments");
        const auto jobs = read_jobs(o.input);
        if (o.validate) { std::cout << json({{"valid_jobs", jobs.size()}, {"model_loaded", false}}).dump() << '\n'; return 0; }
        if (o.model.empty() || o.output.empty()) throw Failure("invalid_arguments");
        if (std::filesystem::exists(o.output)) throw Failure("output_exists");
        if (!std::filesystem::is_regular_file(o.model)) throw Failure("invalid_model");
        const int fd = open(o.output.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd < 0) throw Failure("output_exists_or_unwritable");
        output.reset(fdopen(fd, "w"));
        if (!output) { close(fd); throw Failure("output_io"); }
        llama_log_set(native_log, nullptr);
        // Dynamic backend lookup is independent of ELF library rpath. Load from
        // the exact installed directory, never silently benchmark a CPU fallback.
        ggml_backend_load_all_from_path("/home/hungson175/dev/llama.cpp-prism/build-cuda/bin");
        llama_backend_init();
        auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (o.gpu_layers != 0 && !gpu) throw Failure("gpu_unavailable");
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto model_params = llama_model_default_params(); model_params.n_gpu_layers = o.gpu_layers;
        if (o.gpu_layers != 0) model_params.devices = devices;
        auto start = Clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(o.model.c_str(), model_params), llama_model_free);
        const double load_ms = elapsed(start);
        if (!model) throw Failure("load");
        if (o.gpu_layers != 0 && observed_offloaded_layers <= 0) throw Failure("gpu_offload_unverified");
        if ((o.gpu_layers < 0 || o.gpu_layers >= llama_model_n_layer(model.get()) + 1) &&
            observed_offloaded_layers != observed_total_layers) throw Failure("gpu_offload_incomplete");
        auto params = llama_context_default_params();
        params.n_ctx = o.ctx; params.n_batch = o.batch; params.n_ubatch = o.ubatch; params.n_seq_max = 1;
        params.n_threads = o.threads; params.n_threads_batch = o.threads;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; params.no_perf = false;
        start = Clock::now();
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), params), llama_free);
        const double context_ms = elapsed(start);
        if (!ctx) throw Failure("context");
        const auto * vocab = llama_model_get_vocab(model.get());
        for (int i = 0; i < o.warmup; ++i) {
            auto result = evaluate(ctx.get(), vocab, jobs.front(), o);
            if (!result["ok"].get<bool>()) throw Failure("warmup_distribution");
        }
        write_row(output.get(), {{"type", "metadata"}, {"model_load_ms", load_ms},
            {"context_setup_ms", context_ms}, {"warmup_count", o.warmup}, {"planned", jobs.size()},
            {"gpu_verified", o.gpu_layers != 0 && gpu != nullptr && observed_offloaded_layers > 0},
            {"gpu_device", gpu ? json(ggml_backend_dev_name(gpu)) : json(nullptr)},
            {"gpu_description", gpu ? json(ggml_backend_dev_description(gpu)) : json(nullptr)},
            {"offloaded_layers", observed_offloaded_layers}, {"total_offloadable_layers", observed_total_layers},
            {"full_gpu_offload", observed_offloaded_layers > 0 && observed_offloaded_layers == observed_total_layers},
            {"settings", {{"ctx_size", o.ctx}, {"batch_size", o.batch}, {"ubatch_size", o.ubatch},
                {"threads", o.threads}, {"gpu_layers", o.gpu_layers}, {"flash_attention", true},
                {"n_predict", 1}, {"logit_bias", 50.0}, {"temperature", 1.0}, {"truncation", false},
                {"prompt_add_special", true}, {"mark_add_special", false}, {"parse_special", true},
                {"kv_clear_each_case", true}, {"http", false}, {"missing_policy", "error"}}},
            {"timing_note", "Synchronous prefill plus native sampler; excludes tokenization, reset, load, IO, warmups. No answer-token forward pass."}});
        for (const auto & job : jobs) {
            current_task = job["task_id"].get<std::string>();
            write_row(output.get(), evaluate(ctx.get(), vocab, job, o));
        }
        return 0;
    } catch (const Failure & error) {
        if (output) {
            try { write_row(output.get(), {{"type", "fatal"}, {"task_id", current_task}, {"ok", false}, {"error_kind", error.what()}}); }
            catch (...) {}
        }
        std::cerr << "bonsai-native: " << error.what() << '\n'; return 2;
    } catch (...) { std::cerr << "bonsai-native: internal_error\n"; return 2; }
}
