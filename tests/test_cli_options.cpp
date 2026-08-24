#include "options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ninfer::cli::Options defaults = parse({"ninfer", "model.ninfer", "--prompt", "hello"});
    failures += check(defaults.chat_template_path.empty(),
                      "CLI template override is unexpectedly configured by default");

    const ninfer::cli::Options configured =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--chat-template-file",
               "templates/sharp.jinja"});
    failures += check(configured.chat_template_path == "templates/sharp.jinja",
                      "CLI template override path was not preserved");

    bool empty_path_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--chat-template-file", ""});
    } catch (const std::invalid_argument&) { empty_path_rejected = true; }
    failures += check(empty_path_rejected, "CLI accepted an empty template override path");
    failures += check(ninfer::cli::usage_text("ninfer").find("--chat-template-file") !=
                          std::string::npos,
                      "CLI help omits --chat-template-file");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}