// File errors (CRCError, ReadError, WriteError) must be std::exceptions: the top-level handlers in main(), gpuWorker() and
// Background::run() catch const char*, std::string and std::exception only, so a plain-struct file error escaping to them
// ends in std::terminate (e.g. a full disk while the background thread writes a savefile or proof residue).
#include "../src/File.h"
#include "../src/Background.h"

#include <cstdio>
#include <type_traits>

int main() {
  static_assert(std::is_base_of_v<std::exception, CRCError>);
  static_assert(std::is_base_of_v<std::exception, ReadError>);
  static_assert(std::is_base_of_v<std::exception, WriteError>);

  int failures = 0;
  // The name is kept and what() says what failed.
  try { throw ReadError{"worktodo-0.txt"}; } catch (const std::exception& e) {
    if (std::string(e.what()) != "read error: worktodo-0.txt") { std::printf("FAIL: what() = %s\n", e.what()); ++failures; }
  }
  try { throw CRCError{"proof/42"}; } catch (const CRCError& e) {
    if (e.name != "proof/42") { std::printf("FAIL: name = %s\n", e.name.c_str()); ++failures; }
  }

  // A background task that fails on a full disk is logged, and the background thread keeps running.
  {
    Background background;
    background([] { throw WriteError{"savefile.prp"}; });
    bool ran = false;
    background([&ran] { ran = true; });
    background.waitEmpty();
    if (!ran) { std::printf("FAIL: the background thread stopped after a WriteError\n"); ++failures; }
  }

  if (failures) { return 1; }
  std::printf("file error exception test: ok\n");
  return 0;
}
