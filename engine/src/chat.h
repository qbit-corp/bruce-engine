// ChatML rendering matching the Dory GGUF chat template's common path, plus
// streaming reasoning-tag splitting.
#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "http.h"
#include "json.h"

static const char* kThinkOpen = "<think>";
static const char* kThinkClose = "</think>";

static inline std::string empty_think_pair() {
    return std::string(kThinkOpen) + kThinkClose;
}

static inline std::string chat_content_text(const mj::Value& content) {
    if (content.is_string()) return content.string;
    if (content.is_array()) {
        std::string out;
        for (const auto& part : content.array) {
            if (part.is_string()) {
                out += part.string;
            } else if (part.is_object()) {
                const auto& text = part.at("text");
                if (text.is_string()) out += text.string;
            }
        }
        return out;
    }
    if (content.is_object()) {
        const auto& text = content.at("text");
        if (text.is_string()) return text.string;
    }
    return std::string();
}

static inline std::string tool_arg_string(const mj::Value& v) {
    if (v.is_string()) return v.string;
    if (v.is_object() || v.is_array()) return mj::dump(v);
    if (v.is_null()) return "null";
    if (v.is_bool()) return v.boolean ? "true" : "false";
    if (v.is_number()) return mj::dump(v);
    return std::string();
}

// Options mirrored from the model's Jinja chat template.  The Dory template
// supports enable_thinking, medium_effort and truncate_history_thinking; when
// the latter is true, reasoning is removed from assistant turns that precede
// the most recent user message so old chain-of-thought does not fill the context.
struct ChatTemplateOptions {
    bool enable_thinking = true;
    bool medium_effort = false;
    bool truncate_history_thinking = true;
};

// tool_style mirrors the tool-call branch of the template, which splits on a
// closing tag even when no opening tag is present and always re-adds the empty
// think pair.  The plain assistant branch only truncates when both tags exist.
static inline std::string truncate_thinking_for_history(const std::string& content,
                                                        bool tool_style) {
    std::string c = content;
    if (tool_style) {
        size_t p = c.rfind(kThinkClose);
        if (p != std::string::npos) {
            c = empty_think_pair() + c.substr(p + std::string(kThinkClose).size());
        } else {
            p = c.find(kThinkOpen);
            if (p != std::string::npos) c = c.substr(0, p);
            c = empty_think_pair() + c;
        }
    } else if (c.find(kThinkOpen) != std::string::npos &&
               c.find(kThinkClose) != std::string::npos) {
        size_t p = c.rfind(kThinkClose);
        c = empty_think_pair() + c.substr(p + std::string(kThinkClose).size());
    }
    return c;
}

static inline std::string tool_scalar_text(const mj::Value& v) {
    if (v.is_string()) return v.string;
    if (v.is_null()) return std::string();
    return mj::dump(v);
}

static inline std::string render_tool_extra_keys(const mj::Value& obj,
                                                 std::initializer_list<const char*> handled) {
    std::string out;
    if (!obj.is_object()) return out;
    for (const auto& kv : obj.object) {
        bool skip = false;
        for (const char* h : handled) {
            if (kv.first == h) { skip = true; break; }
        }
        if (skip) continue;
        const mj::Value& v = kv.second;
        out += "\n<" + kv.first + ">";
        out += (v.is_object() || v.is_array()) ? mj::dump(v) : tool_scalar_text(v);
        out += "</" + kv.first + ">";
    }
    return out;
}

// Renders the tools section from the Dory GGUF chat template.  The model was
// trained on this exact XML layout, so keep the wording and whitespace aligned.
static inline std::string render_tools_prompt(const mj::Value& tools) {
    if (!tools.is_array() || tools.array.empty()) return std::string();
    std::string out = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto& entry : tools.array) {
        if (!entry.is_object()) continue;
        const mj::Value& fn = entry.at("function").is_object() ? entry.at("function") : entry;
        out += "\n<function>\n<name>" + fn.at("name").as_string() + "</name>";
        if (fn.has("description")) {
            out += "\n<description>" + trim_str(fn.at("description").as_string()) + "</description>";
        }
        out += "\n<parameters>";
        const mj::Value& params = fn.at("parameters");
        if (params.is_object()) {
            const mj::Value& props = params.at("properties");
            if (props.is_object()) {
                for (const auto& p : props.object) {
                    const mj::Value& f = p.second;
                    out += "\n<parameter>\n<name>" + p.first + "</name>";
                    if (f.is_object() && f.has("type")) {
                        out += "\n<type>" + tool_scalar_text(f.at("type")) + "</type>";
                    }
                    if (f.is_object() && f.has("description")) {
                        out += "\n<description>" + trim_str(f.at("description").as_string()) + "</description>";
                    }
                    if (f.is_object() && f.has("enum")) {
                        out += "\n<enum>" + mj::dump(f.at("enum")) + "</enum>";
                    }
                    if (f.is_object()) {
                        out += render_tool_extra_keys(f, {"name", "type", "description", "enum"});
                    }
                    out += "\n</parameter>";
                }
            }
            out += render_tool_extra_keys(params, {"type", "properties", "required"});
            if (params.has("required")) {
                out += "\n<required>" + mj::dump(params.at("required")) + "</required>";
            }
        }
        out += "\n</parameters>";
        out += render_tool_extra_keys(fn, {"type", "name", "description", "parameters"});
        out += "\n</function>";
    }
    out += "\n</tools>";
    out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
           "<tool_call>\n<function=example_function_name>\n"
           "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
           "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
           "</function>\n</tool_call>\n\n"
           "<IMPORTANT>\nReminder:\n"
           "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
           "- Required parameters MUST be specified\n"
           "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
           "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
           "</IMPORTANT>";
    return out;
}

static inline std::string render_chat_prompt(const mj::Value& messages,
                                             const ChatTemplateOptions& opts,
                                             bool add_generation_prompt = true,
                                             const mj::Value* tools = nullptr) {
    std::string out;
    if (!messages.is_array() || messages.array.empty()) return out;

    const std::string tools_text = tools ? render_tools_prompt(*tools) : std::string();
    size_t start = 0;
    const mj::Value& first = messages.array[0];
    const std::string first_role = first.is_object() ? first.at("role").as_string() : std::string();
    if (first_role == "system" || first_role == "developer") {
        const std::string sys = chat_content_text(first.at("content"));
        out += "<|im_start|>system\n";
        out += sys;
        if (!tools_text.empty()) {
            if (!sys.empty()) out += "\n\n";
            out += tools_text;
        }
        out += "<|im_end|>\n";
        start = 1;
    } else if (!tools_text.empty()) {
        out += "<|im_start|>system\n" + tools_text + "<|im_end|>\n";
    }

    // The template computes this index over messages after the optional system
    // message, and compares it with loop.index0 of the same list.
    int last_user_idx = -1;
    for (size_t i = start; i < messages.array.size(); ++i) {
        const mj::Value& m = messages.array[i];
        if (m.is_object() && m.at("role").as_string() == "user") {
            last_user_idx = (int)(i - start);
        }
    }

    for (size_t i = start; i < messages.array.size(); ++i) {
        const mj::Value& m = messages.array[i];
        if (!m.is_object()) continue;
        const int li = (int)(i - start);
        std::string role = m.at("role").as_string("user");
        std::string content = chat_content_text(m.at("content"));

        if (role == "assistant") {
            const mj::Value& rc = m.at("reasoning_content");
            if (rc.is_string() && !trim_str(rc.string).empty()) {
                content = std::string(kThinkOpen) + "\n" + rc.string + kThinkClose + content;
            } else if (content.find(kThinkOpen) == std::string::npos &&
                       content.find(kThinkClose) == std::string::npos) {
                content = empty_think_pair() + content;
            }

            const bool truncate = opts.truncate_history_thinking && li < last_user_idx;
            const mj::Value& calls = m.at("tool_calls");
            if (calls.is_array() && !calls.array.empty()) {
                out += "<|im_start|>assistant\n";
                if (!trim_str(content).empty()) {
                    if (!truncate) {
                        out += trim_str(content) + "\n";
                    } else {
                        out += truncate_thinking_for_history(content, true) + "\n";
                    }
                } else {
                    out += empty_think_pair();
                }
                for (const auto& call : calls.array) {
                    if (!call.is_object()) continue;
                    const mj::Value& fn = call.at("function").is_object() ? call.at("function") : call;
                    std::string name = fn.at("name").as_string();
                    out += "<tool_call>\n<function=" + name + ">\n";
                    const mj::Value& args = fn.at("arguments");
                    mj::Value parsed;
                    bool has_args = false;
                    if (args.is_object()) {
                        parsed = args;
                        has_args = true;
                    } else if (args.is_string()) {
                        try {
                            parsed = mj::parse(args.string);
                            has_args = parsed.is_object();
                        } catch (...) {
                            has_args = false;
                        }
                    }
                    if (has_args) {
                        for (const auto& kv : parsed.object) {
                            out += "<parameter=" + kv.first + ">\n";
                            out += tool_arg_string(kv.second);
                            out += "\n</parameter>\n";
                        }
                    }
                    out += "</function>\n</tool_call>\n";
                }
                out += "<|im_end|>\n";
            } else {
                if (!truncate) {
                    out += "<|im_start|>assistant\n";
                    out += trim_str(content);
                    out += "<|im_end|>\n";
                } else {
                    std::string c = truncate_thinking_for_history(content, false);
                    c = trim_str(c);
                    if (!c.empty()) {
                        out += "<|im_start|>assistant\n" + c + "<|im_end|>\n";
                    } else {
                        out += "<|im_start|>assistant\n<|im_end|>\n";
                    }
                }
            }
        } else if (role == "user" || role == "system" || role == "developer") {
            out += "<|im_start|>" + std::string(role == "developer" ? "system" : role) + "\n";
            if (role == "user" && li == last_user_idx && opts.medium_effort) {
                out += content + "\n\n{reasoning effort: efficient}";
            } else {
                out += content;
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            bool prev_tool = i > start && messages.array[i - 1].is_object() &&
                             messages.array[i - 1].at("role").as_string() == "tool";
            if (!prev_tool) out += "<|im_start|>user\n";
            out += "<tool_response>\n" + content + "\n</tool_response>\n";
            bool next_tool = i + 1 < messages.array.size() && messages.array[i + 1].is_object() &&
                             messages.array[i + 1].at("role").as_string() == "tool";
            if (!next_tool) out += "<|im_end|>\n";
        } else {
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
    }

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        out += opts.enable_thinking ? (std::string(kThinkOpen) + "\n")
                                    : (std::string(kThinkOpen) + kThinkClose);
    }
    return out;
}

static inline std::string render_chat_prompt(const mj::Value& messages, bool enable_thinking,
                                             bool add_generation_prompt = true) {
    ChatTemplateOptions opts;
    opts.enable_thinking = enable_thinking;
    return render_chat_prompt(messages, opts, add_generation_prompt);
}

struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

static inline std::string strip_tool_value_newlines(const std::string& s) {
    size_t a = 0, b = s.size();
    if (a < b && s[a] == '\r') ++a;
    if (a < b && s[a] == '\n') ++a;
    if (b > a && s[b - 1] == '\n') --b;
    if (b > a && s[b - 1] == '\r') --b;
    return s.substr(a, b - a);
}

static inline std::string tool_schema_type(const mj::Value& prop) {
    if (!prop.is_object()) return std::string();
    const mj::Value& t = prop.at("type");
    if (t.is_string()) return t.string;
    if (t.is_array()) {
        for (const auto& e : t.array) {
            if (e.is_string() && e.string != "null") return e.string;
        }
    }
    return std::string();
}

static inline const mj::Value* find_tool_schema(const mj::Value& tools, const std::string& name) {
    if (!tools.is_array()) return nullptr;
    for (const auto& entry : tools.array) {
        if (!entry.is_object()) continue;
        const mj::Value& fn = entry.at("function").is_object() ? entry.at("function") : entry;
        if (fn.at("name").as_string() == name) return &fn;
    }
    return nullptr;
}

static inline const mj::Value* find_tool_param(const mj::Value& tools, const std::string& fn_name,
                                               const std::string& param) {
    const mj::Value* fn = find_tool_schema(tools, fn_name);
    if (!fn) return nullptr;
    const mj::Value& props = fn->at("parameters").at("properties");
    if (!props.is_object()) return nullptr;
    for (const auto& kv : props.object) {
        if (kv.first == param) return &kv.second;
    }
    return nullptr;
}

static inline mj::Value coerce_tool_value(const std::string& raw, const std::string& type) {
    if (type.empty() || type == "string") return mj::Value(raw);
    const std::string t = trim_str(raw);
    if (t.empty()) return mj::Value(raw);
    if (type == "integer") {
        char* end = nullptr;
        long long v = std::strtoll(t.c_str(), &end, 10);
        if (end && *end == '\0') return mj::Value((int64_t)v);
    } else if (type == "number") {
        char* end = nullptr;
        double v = std::strtod(t.c_str(), &end);
        if (end && *end == '\0') return mj::Value(v);
    } else if (type == "boolean") {
        std::string lower = t;
        for (char& c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower == "true") return mj::Value(true);
        if (lower == "false") return mj::Value(false);
    } else if (type == "object" || type == "array") {
        try { return mj::parse(t); } catch (...) {}
    }
    return mj::Value(raw);
}

static inline bool parse_tool_block(const std::string& block, const mj::Value& tools,
                                    std::string& name, std::string& args_json) {
    const size_t fn_tag = block.find("<function=");
    if (fn_tag == std::string::npos) {
        try {
            mj::Value v = mj::parse(trim_str(block));
            if (!v.is_object()) return false;
            const mj::Value& fn = v.at("function").is_object() ? v.at("function") : v;
            name = trim_str(fn.at("name").as_string());
            const mj::Value& a = fn.at("arguments");
            if (a.is_object() || a.is_array()) {
                args_json = mj::dump(a);
            } else if (a.is_string()) {
                const std::string s = trim_str(a.string);
                try {
                    (void)mj::parse(s);
                    args_json = s;
                } catch (...) {
                    mj::Value o = mj::Value::obj();
                    o["value"] = a.string;
                    args_json = mj::dump(o);
                }
            } else if (a.is_null()) {
                args_json = "{}";
            }
            return !name.empty();
        } catch (...) {
            return false;
        }
    }

    const size_t name_start = fn_tag + std::strlen("<function=");
    const size_t name_end = block.find('>', name_start);
    if (name_end == std::string::npos) return false;
    name = trim_str(block.substr(name_start, name_end - name_start));
    if (name.empty()) return false;

    mj::Value args = mj::Value::obj();
    const size_t ptag = std::strlen("<parameter=");
    const size_t pclose = std::strlen("</parameter>");
    size_t pos = name_end + 1;
    while (true) {
        size_t p = block.find("<parameter=", pos);
        if (p == std::string::npos) break;
        size_t ps = p + ptag;
        size_t pe = block.find('>', ps);
        if (pe == std::string::npos) return false;
        std::string pname = trim_str(block.substr(ps, pe - ps));
        size_t ve = block.find("</parameter>", pe + 1);
        if (ve == std::string::npos) return false;
        std::string raw = strip_tool_value_newlines(block.substr(pe + 1, ve - (pe + 1)));
        const mj::Value* prop = find_tool_param(tools, name, pname);
        std::string type = prop ? tool_schema_type(*prop) : std::string();
        if (!pname.empty()) args[pname] = coerce_tool_value(raw, type);
        pos = ve + pclose;
    }
    args_json = mj::dump(args);
    return true;
}

// Extracts <tool_call> blocks.  Text outside the calls stays in content so a
// short preamble is not lost.
static inline std::vector<ParsedToolCall> parse_tool_calls(const std::string& text,
                                                           std::string& content,
                                                           const mj::Value& tools) {
    static const std::string open = "<tool_call>";
    static const std::string close = "</tool_call>";
    std::vector<ParsedToolCall> calls;
    size_t pos = 0;
    while (true) {
        size_t s = text.find(open, pos);
        if (s == std::string::npos) break;
        size_t e = text.find(close, s + open.size());
        if (e == std::string::npos) break;
        std::string name, args;
        if (parse_tool_block(text.substr(s + open.size(), e - (s + open.size())), tools,
                             name, args)) {
            ParsedToolCall c;
            c.name = std::move(name);
            c.arguments = std::move(args);
            calls.push_back(std::move(c));
        }
        pos = e + close.size();
    }
    if (calls.empty()) {
        content = text;
        return calls;
    }
    size_t first = text.find(open);
    size_t last = text.rfind(close);
    std::string before, after;
    if (first != std::string::npos) before = text.substr(0, first);
    if (last != std::string::npos) after = text.substr(last + close.size());
    before = trim_str(before);
    after = trim_str(after);
    content = before;
    if (!content.empty() && !after.empty()) content += "\n";
    content += after;
    content = trim_str(content);
    return calls;
}

static inline std::string simple_chat_prompt(const std::string& user, bool enable_thinking) {
    std::string out = "<|im_start|>user\n" + user + "<|im_end|>\n";
    out += "<|im_start|>assistant\n";
    out += enable_thinking ? (std::string(kThinkOpen) + "\n") : (std::string(kThinkOpen) + kThinkClose);
    return out;
}

static inline size_t match_leading_think_tag(const std::string& s, const char* core) {
    const size_t n = std::strlen(core);
    if (s.size() >= n && s.compare(0, n, core) == 0) return n;
    if (s.size() >= n + 1 && s[0] == ' ' && s.compare(1, n, core) == 0) return n + 1;
    return 0;
}

static inline void trim_leading_space(std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    s.erase(0, i);
}

static inline void trim_trailing_space(std::string& s) {
    size_t n = s.size();
    while (n > 0 && std::isspace((unsigned char)s[n - 1])) --n;
    s.resize(n);
}

// True when the generation prompt already opened a think block, so the model's
// output starts inside it.  The Dory template does this for both thinking modes.
static inline bool prompt_opens_thinking(const std::string& prompt) {
    std::string open = kThinkOpen;
    if (prompt.size() >= open.size() &&
        prompt.compare(prompt.size() - open.size(), open.size(), open) == 0) return true;
    open.push_back('\n');
    return prompt.size() >= open.size() &&
           prompt.compare(prompt.size() - open.size(), open.size(), open) == 0;
}

static inline void split_think_text(const std::string& text, std::string& reasoning,
                                    std::string& content, bool thinking_prompt = false) {
    reasoning.clear();
    content = text;
    trim_leading_space(content);
    bool in_reasoning = thinking_prompt;

    // The model may echo the empty think pair at the start of the answer.
    size_t lead = match_leading_think_tag(content, kThinkClose);
    if (lead > 0) {
        content.erase(0, lead);
        trim_leading_space(content);
        return;
    }
    lead = match_leading_think_tag(content, kThinkOpen);
    if (lead > 0) {
        content.erase(0, lead);
        trim_leading_space(content);
        in_reasoning = true;
    }

    const size_t tag_len = std::strlen(kThinkClose);
    size_t p = content.find(kThinkClose);
    if (p != std::string::npos) {
        std::string before = content.substr(0, p);
        std::string after = content.substr(p + tag_len);
        trim_leading_space(after);
        if (in_reasoning) {
            reasoning = trim_str(before);
            content = std::move(after);
        } else {
            content = before + after;
        }
    } else if (in_reasoning) {
        // Generation was cut off before the model closed its reasoning block.
        reasoning = trim_str(content);
        content.clear();
    }
}

// Splits generated text into reasoning_content / content while streaming.  When the
// generation prompt ends with an open think tag, the stream starts in reasoning mode.
struct ThinkSplitter {
    bool thinking = false;
    bool at_start = true;
    bool leading_trim = true;
    std::string buf;
    std::function<bool(const std::string&, bool)> emit;

    bool emit_text(std::string text, bool reasoning) {
        if (leading_trim) {
            trim_leading_space(text);
            if (text.empty()) return true;
            leading_trim = false;
        }
        if (text.empty()) return true;
        return emit(text, reasoning);
    }

    void feed(const std::string& text) {
        buf += text;
        process(false);
    }

    void process(bool flush) {
        const size_t tag_len = std::strlen(kThinkClose);
        while (true) {
            if (thinking && at_start) {
                size_t lead = match_leading_think_tag(buf, kThinkClose);
                if (lead > 0) {
                    buf.erase(0, lead);
                    thinking = false;
                    at_start = false;
                    leading_trim = true;
                    continue;
                }
                lead = match_leading_think_tag(buf, kThinkOpen);
                if (lead > 0) {
                    buf.erase(0, lead);
                    at_start = false;
                    leading_trim = true;
                    continue;
                }
                // Wait until the buffer cannot be the start of a tag anymore.
                if (!flush && buf.size() < tag_len + 1) return;
                at_start = false;
            }
            size_t p = buf.find(kThinkClose);
            if (p != std::string::npos) {
                std::string before = buf.substr(0, p);
                if (thinking) trim_trailing_space(before);
                if (!emit_text(std::move(before), thinking)) {
                    buf.clear();
                    return;
                }
                buf.erase(0, p + tag_len);
                thinking = false;
                leading_trim = true;
                continue;
            }
            if (flush) {
                std::string out = std::move(buf);
                buf.clear();
                if (thinking) trim_trailing_space(out);
                emit_text(std::move(out), thinking);
                return;
            }
            if (buf.size() > tag_len - 1) {
                size_t n = buf.size() - (tag_len - 1);
                std::string out = buf.substr(0, n);
                buf.erase(0, n);
                if (thinking && !leading_trim) {
                    // Keep trailing whitespace until we know it is not before the tag.
                    size_t e = out.size();
                    while (e > 0 && std::isspace((unsigned char)out[e - 1])) --e;
                    if (e < out.size()) {
                        buf.insert(0, out.substr(e));
                        out.resize(e);
                    }
                }
                if (!emit_text(std::move(out), thinking)) {
                    buf.clear();
                    return;
                }
            }
            return;
        }
    }

    void finish() {
        process(true);
    }
};
