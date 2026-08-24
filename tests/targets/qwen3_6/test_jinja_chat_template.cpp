#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

fi::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(text)));
    return message;
}

} // namespace

int main() {
    constexpr std::string_view source = R"JINJA(
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for item in content -%}
            {%- if item.type == 'text' -%}{{- item.text -}}
            {%- elif item.type == 'image' -%}{{- '<image>' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- set state = namespace(matched=false) -%}
{%- set chained = '<think>\nreason\n</think>'.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}
{%- for message in messages -%}
    {%- set content = render_content(message.content).lstrip().rstrip() -%}
    {%- if message.role == 'user' and content.startswith('hello') and content.endswith('world') -%}
        {%- set state.matched = true -%}
    {%- endif -%}
    {{- message.role ~ ':' ~ content ~ '/' ~ (content.split(' ') | length) ~ ';' -}}
    {%- if message.tool_calls -%}
        {{- 'args=' ~ (message.tool_calls[0].function.arguments | tojson) ~ ';' -}}
    {%- endif -%}
{%- endfor -%}
{{- 'matched=' ~ state.matched ~ ';tools=' ~ (tools | length) ~ ';preserve=' ~ chat_template_kwargs.preserve_thinking ~ ';chain=' ~ chained ~ ';effort=' ~ reasoning_effort -}}
)JINJA";

    int failures = 0;
    const fi::CompiledChatTemplate chat_template =
        fi::CompiledChatTemplate::compile_jinja(std::string(source), "test-template");
    const ninfer::PromptCapabilities capabilities = chat_template.capabilities();
    failures += check(capabilities.enable_thinking,
                      "custom Jinja template did not expose thinking control");
    failures += check(capabilities.reasoning_effort.low && capabilities.reasoning_effort.medium &&
                          capabilities.reasoning_effort.xhigh &&
                          capabilities.reasoning_effort.default_effort ==
                              ninfer::ReasoningEffort::Medium,
                      "custom Jinja template did not expose reasoning-effort presets");

    fi::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(fi::ChatPart::text_part(" hello "));
    user.parts.push_back(fi::ChatPart::image({}));
    user.parts.push_back(fi::ChatPart::text_part("world "));

    fi::ChatMessage assistant = text_message(ninfer::ChatRole::Assistant, "done");
    assistant.tool_calls.push_back(
        fi::ToolCall{.id = "call-1", .name = "lookup", .arguments_json = R"({"query":"qwen"})"});

    fi::ChatRenderOptions options;
    options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    options.preserve_thinking = true;
    options.tool_jsons.push_back(R"({"type":"function","function":{"name":"lookup"}})");
    const fi::RenderedChat rendered = chat_template.render({std::move(user), std::move(assistant)},
                                                            std::move(options));
    const std::string expected =
        "user:hello <image>world/2;assistant:done/1;args={\"query\": \"qwen\"};"
        "matched=True;tools=1;preserve=True;chain=reason;effort=medium";
    failures += check(rendered.text == expected,
                      ("custom Jinja template context rendered unexpected prompt text: " +
                       rendered.text)
                          .c_str());
    failures += check(!rendered.rewrite_checkpoint.has_value(),
                      "custom Jinja template unexpectedly exposed a rewrite checkpoint");

    bool malformed_rejected = false;
    try {
        (void)fi::CompiledChatTemplate::compile_jinja("{% if", "malformed-template");
    } catch (const std::invalid_argument&) { malformed_rejected = true; }
    failures += check(malformed_rejected, "malformed Jinja template was accepted");

    const fi::CompiledChatTemplate no_effort_template = fi::CompiledChatTemplate::compile_jinja(
        "{% for message in messages %}{{ message.content }}{% endfor %}", "no-effort-template");
    bool effort_rejected = false;
    try {
        fi::ChatRenderOptions effort_options;
        effort_options.reasoning_effort = ninfer::ReasoningEffort::Low;
        (void)no_effort_template.render({text_message(ninfer::ChatRole::User, "hello")},
                                        std::move(effort_options));
    } catch (const std::invalid_argument&) { effort_rejected = true; }
    failures += check(effort_rejected,
                      "Jinja template without reasoning_effort accepted reasoning effort");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}