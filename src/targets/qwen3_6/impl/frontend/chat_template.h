#pragma once

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
    [[nodiscard]] std::string rendered_content(bool add_vision_id = false,
                                               int* image_count   = nullptr,
                                               int* video_count   = nullptr) const;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
};

// True when the rendered prompt leaves the reasoning channel open: the final
// thinking-open marker has no subsequent thinking-close marker. Used to seed
// `starts_in_reasoning` from the actually-rendered prompt rather than from
// render options alone.
[[nodiscard]] bool prompt_ends_in_open_reasoning(const std::string& rendered_text);

enum class ChatTemplateSemantics : std::uint8_t {
    ThinkingToggle,
    ReasoningEffort,
};

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate resolve(std::string_view source);
    [[nodiscard]] static CompiledChatTemplate compile_jinja(std::string source,
                                                             std::string source_name);

    [[nodiscard]] PromptCapabilities capabilities() const noexcept;
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options = {}) const;

private:
    class JinjaTemplate;

    explicit CompiledChatTemplate(ChatTemplateSemantics semantics) noexcept
        : semantics_(semantics) {}
    explicit CompiledChatTemplate(std::shared_ptr<const JinjaTemplate> jinja_template) noexcept
        : jinja_template_(std::move(jinja_template)) {}

    ChatTemplateSemantics semantics_ = ChatTemplateSemantics::ThinkingToggle;
    std::shared_ptr<const JinjaTemplate> jinja_template_;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
