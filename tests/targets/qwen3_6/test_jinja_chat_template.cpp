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
    failures += check(rendered.media_placeholders.empty(),
                      "jinja template without pad markers recorded placeholders");

    // The Jinja render path must record structured media placeholders: one per
    // media part, in request order, at the exact pad-marker bytes.
    const std::string image_marker = "\x3c\x7cimage_pad\x7c\x3e";
    const std::string video_marker = "\x3c\x7cvideo_pad\x7c\x3e";
    std::string vision_source = std::string(
        "{% for m in messages %}{% for item in m.content %}"
        "{% if item.type == 'image' %}<|vision_start|>" + image_marker +
        "<|vision_end|>"
        "{% elif item.type == 'video' %}<|vision_start|>" + video_marker +
        "<|vision_end|>"
        "{% elif item.type == 'text' %}{{ item.text }}"
        "{% endif %}{% endfor %}{% endfor %}");
    const fi::CompiledChatTemplate vision_template =
        fi::CompiledChatTemplate::compile_jinja(vision_source, "vision-template");

    fi::ChatMessage vision_user;
    vision_user.role = ninfer::ChatRole::User;
    vision_user.parts.push_back(fi::ChatPart::text_part("a "));
    vision_user.parts.push_back(fi::ChatPart::image({}));
    vision_user.parts.push_back(fi::ChatPart::video({}));
    vision_user.parts.push_back(fi::ChatPart::text_part(" c"));
    const fi::RenderedChat vision_rendered =
        vision_template.render({vision_user}, fi::ChatRenderOptions{});
    failures += check(vision_rendered.media_placeholders.size() == 2,
                      "jinja render did not record one placeholder per media part");
    if (vision_rendered.media_placeholders.size() == 2) {
        const fi::MediaPlaceholderByteSpec& image = vision_rendered.media_placeholders[0];
        const fi::MediaPlaceholderByteSpec& video = vision_rendered.media_placeholders[1];
        const auto span_text = [&](const fi::MediaPlaceholderByteSpec& spec) {
            return vision_rendered.text.substr(spec.bytes.begin,
                                                spec.bytes.end - spec.bytes.begin);
        };
        failures += check(image.modality == fi::Modality::Image && image.item_index == 0,
                          "first jinja placeholder is not the image part");
        failures += check(video.modality == fi::Modality::Video && video.item_index == 1,
                          "second jinja placeholder is not the video part");
        failures += check(span_text(image) == image_marker,
                          "image placeholder bytes do not cover the image pad marker");
        failures += check(span_text(video) == video_marker,
                          "video placeholder bytes do not cover the video pad marker");
    }

    // Literal pad markers quoted in text are escaped upstream, not recorded.
    fi::ChatMessage literal_user;
    literal_user.role = ninfer::ChatRole::User;
    literal_user.parts.push_back(fi::ChatPart::text_part("quote " + image_marker + " here"));
    literal_user.parts.push_back(fi::ChatPart::image({}));
    const fi::RenderedChat literal_rendered =
        vision_template.render({literal_user}, fi::ChatRenderOptions{});
    const std::string broken_marker = "<|\xE2\x81\xA0image_pad|>";
    failures += check(literal_rendered.media_placeholders.size() == 1,
                      "literal pad marker in text was not escaped before scanning");
    failures += check(literal_rendered.text.find(broken_marker) != std::string::npos,
                      "literal pad marker was not broken in the rendered text");

    // Poisoned history (2026-08-30 t11v incident): plain markers quoted in a text
    // part, an assistant reasoning channel, a tool-call arguments JSON, and a tool
    // result are all escaped upstream; only the structured parts' pads are
    // recorded, in request order.
    std::string poisoned_source = std::string(
        "{% for m in messages %}{% for item in m.content %}"
        "{% if item.type == 'image' %}IMG" + image_marker +
        "{% elif item.type == 'video' %}VID" + video_marker +
        "{% elif item.type == 'text' %}{{ item.text }}"
        "{% endif %}{% endfor %}"
        "{% if m.tool_calls %}ARGS{{ m.tool_calls[0].function.arguments | tojson }}"
        "ENDARGS{% endif %}"
        "{% if m.reasoning_content %}REASON{{ m.reasoning_content }}ENDREASON{% endif %}"
        "{% endfor %}");
    const fi::CompiledChatTemplate poisoned_template =
        fi::CompiledChatTemplate::compile_jinja(poisoned_source, "poisoned-template");

    fi::ChatMessage poisoned_user;
    poisoned_user.role = ninfer::ChatRole::User;
    poisoned_user.parts.push_back(fi::ChatPart::text_part("a "));
    poisoned_user.parts.push_back(fi::ChatPart::image({}));
    poisoned_user.parts.push_back(fi::ChatPart::image({}));
    poisoned_user.parts.push_back(
        fi::ChatPart::text_part("quoted " + std::string(image_marker) + " here"));
    fi::ChatMessage poisoned_tools;
    poisoned_tools.role = ninfer::ChatRole::Assistant;
    poisoned_tools.reasoning_content =
        "debugging the spelling of " + std::string(image_marker) + " now";
    const std::string poisoned_args =
        std::string(R"({"path":"tpl.jinja","content":"line )") + std::string(image_marker) +
        std::string("\"}");
    poisoned_tools.tool_calls.push_back(
        fi::ToolCall{.id = "call-poison", .name = "write", .arguments_json = poisoned_args});
    fi::ChatMessage poisoned_result;
    poisoned_result.role         = ninfer::ChatRole::Tool;
    poisoned_result.tool_call_id = "call-poison";
    poisoned_result.parts.push_back(
        fi::ChatPart::text_part("wrote " + std::string(image_marker)));
    const fi::RenderedChat poisoned_rendered = poisoned_template.render(
        {poisoned_user, poisoned_tools, poisoned_result}, fi::ChatRenderOptions{});
    failures += check(poisoned_rendered.media_placeholders.size() == 2,
                      "poisoned history recorded more placeholders than structured parts");
    if (poisoned_rendered.media_placeholders.size() == 2) {
        const auto span_text = [&](const fi::MediaPlaceholderByteSpec& spec) {
            return poisoned_rendered.text.substr(spec.bytes.begin,
                                                 spec.bytes.end - spec.bytes.begin);
        };
        failures += check(
            poisoned_rendered.media_placeholders[0].modality == fi::Modality::Image &&
                poisoned_rendered.media_placeholders[0].item_index == 0 &&
                poisoned_rendered.media_placeholders[1].modality == fi::Modality::Image &&
                poisoned_rendered.media_placeholders[1].item_index == 1 &&
                span_text(poisoned_rendered.media_placeholders[0]) == image_marker &&
                span_text(poisoned_rendered.media_placeholders[1]) == image_marker,
            "poisoned history placeholders do not cover the structured pads in order");
        std::size_t plain_count = 0;
        for (std::size_t pos = poisoned_rendered.text.find(image_marker);
             pos != std::string::npos; pos = poisoned_rendered.text.find(image_marker, pos + 1)) {
            ++plain_count;
        }
        failures += check(plain_count == 2,
                          "poisoned history rendered text still contains stray plain markers");
        const std::size_t args_begin = poisoned_rendered.text.find("ARGS");
        const std::size_t args_end   = poisoned_rendered.text.find("ENDARGS");
        failures += check(args_begin != std::string::npos && args_end > args_begin,
                          "poisoned history render missing tool-call arguments region");
        if (args_begin != std::string::npos && args_end > args_begin) {
            const std::string args_region = poisoned_rendered.text.substr(args_begin,
                                                                          args_end - args_begin);
            failures += check(args_region.find(image_marker) == std::string::npos,
                              "tool-call arguments leaked a plain pad marker into the render");
            failures += check(args_region.find(broken_marker) != std::string::npos,
                              "tool-call arguments pad marker was not escaped");
        }
    }

    // Count cap: a plain marker outside the structured parts (prose after the
    // last part) is never recorded, even though it appears in the rendered
    // text - the per-modality cap ignores excess markers in text order.
    std::string cap_source = std::string(
        "{% for m in messages %}{% for item in m.content %}"
        "{% if item.type == 'image' %}IMG" + image_marker +
        "{% elif item.type == 'text' %}{{ item.text }}"
        "{% endif %}{% endfor %}{% endfor %}"
        "TRAILER" + image_marker);
    const fi::CompiledChatTemplate cap_template =
        fi::CompiledChatTemplate::compile_jinja(cap_source, "cap-template");
    fi::ChatMessage cap_user;
    cap_user.role = ninfer::ChatRole::User;
    cap_user.parts.push_back(fi::ChatPart::image({}));
    const fi::RenderedChat cap_rendered = cap_template.render({cap_user}, fi::ChatRenderOptions{});
    failures += check(cap_rendered.media_placeholders.size() == 1,
                      "jinja render recorded a stray plain marker despite the part count cap");
    if (cap_rendered.media_placeholders.size() == 1) {
        const fi::MediaPlaceholderByteSpec& spec = cap_rendered.media_placeholders[0];
        failures += check(
            cap_rendered.text.substr(spec.bytes.begin,
                                     spec.bytes.end - spec.bytes.begin) == image_marker &&
                spec.bytes.begin >= 3 &&
                cap_rendered.text.substr(spec.bytes.begin - 3, 3) == "IMG",
            "count cap recorded the stray marker instead of the structured pad");
    }

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