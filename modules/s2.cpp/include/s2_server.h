#pragma once

#include "httplib.h"
#include "s2_audio.h"
#include "s2_pipeline.h"

#include <cstdint>
#include <string>

namespace s2
{

  struct ServerParams
  {
    std::string host = "127.0.0.1";
    int32_t port = 3030;

    PipelineParams pipeline;
  };

  class Server
  {
  public:
    Server();
    ~Server();

    bool serve(const ServerParams& params);
  };

}
