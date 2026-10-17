#pragma once

#include "PtxRegisterCache.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

struct AevumSmemSpillMarker {
  std::string compileArgs;
  bool enabled{false};
};

inline AevumSmemSpillMarker aevumTakeSmemSpillMarker(const std::string& args) {
  AevumSmemSpillMarker result;
  std::istringstream input(args);
  for (std::string token; input >> token;) {
    if (token.rfind("--aevum-smemspill", 0) == 0) {
      if (token != "--aevum-smemspill=1" || result.enabled)
        throw std::runtime_error("invalid or duplicate shared-spill marker");
      result.enabled = true;
      continue;
    }
    result.compileArgs += token + ' ';
  }
  return result;
}

inline std::string aevumPtxWithSmemSpill(const std::string& ptx,
                                         const std::string& name) {
  using namespace aevum_ptx_detail;
  if (name.empty()) throw std::runtime_error("empty shared-spill kernel name");

  const std::string code = codeOnly(ptx);
  const size_t version = code.find(".version ");
  if (version == std::string::npos)
    throw std::runtime_error("shared spilling requires a PTX version");
  std::istringstream vs(code.substr(version + 9));
  unsigned major = 0, minor = 0;
  char dot = 0;
  if (!(vs >> major >> dot >> minor) || dot != '.' || major < 9)
    throw std::runtime_error("shared spilling requires PTX 9.0+");

  size_t entryEnd = std::string::npos;
  for (size_t pos = 0; (pos = code.find(".entry", pos)) != std::string::npos; pos += 6) {
    if ((pos && identifier(code[pos - 1])) ||
        pos + 6 == code.size() || !space(code[pos + 6]))
      continue;
    size_t start = pos + 6;
    while (start < code.size() && space(code[start])) ++start;
    size_t end = start;
    while (end < code.size() && identifier(code[end])) ++end;
    if (code.substr(start, end - start) != name) continue;
    if (entryEnd != std::string::npos)
      throw std::runtime_error("duplicate named PTX entry");
    entryEnd = end;
  }
  if (entryEnd == std::string::npos)
    throw std::runtime_error("named PTX entry not found");

  size_t params = entryEnd;
  while (params < code.size() && space(code[params])) ++params;
  if (params == code.size() || code[params] != '(')
    throw std::runtime_error("malformed PTX entry parameters");

  unsigned depth = 1;
  size_t cur = params + 1;
  for (; cur < code.size() && depth; ++cur) {
    if (code[cur] == '(') ++depth;
    else if (code[cur] == ')') --depth;
  }
  if (depth) throw std::runtime_error("unterminated PTX entry parameters");

  const size_t body = code.find('{', cur);
  if (body == std::string::npos)
    throw std::runtime_error("missing PTX entry body");
  const std::string header = code.substr(cur, body - cur);
  if (header.find(".entry") != std::string::npos ||
      header.find(".func") != std::string::npos)
    throw std::runtime_error("malformed PTX entry header");
  if (header.find(".maxntid") == std::string::npos)
    throw std::runtime_error("shared spilling requires .maxntid");

  size_t bodyEnd = body + 1;
  depth = 1;
  for (; bodyEnd < code.size() && depth; ++bodyEnd) {
    if (code[bodyEnd] == '{') ++depth;
    else if (code[bodyEnd] == '}') --depth;
  }
  if (depth) throw std::runtime_error("unterminated PTX entry body");

  const std::string bodyText = ptx.substr(body + 1, bodyEnd - body - 2);
  if (bodyText.find("enable_smem_spilling") != std::string::npos)
    return ptx;

  std::string result = ptx;
  result.insert(body + 1, "\n.pragma \"enable_smem_spilling\";");
  return result;
}
