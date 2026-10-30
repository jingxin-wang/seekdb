/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <limits>
#include <string>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

namespace oceanbase
{
namespace common
{
namespace
{
class ProviderTest : public ::testing::Test
{
protected:
  ObJsonObject *parse(const char *json)
  {
    ObString text = ObString::make_string(json);
    ObJsonObject *result = nullptr;
    EXPECT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_object_form_str(allocator_, text, result));
    return result;
  }
  ObArenaAllocator allocator_;
};

TEST_F(ProviderTest, openai_embeddings_restore_input_order)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(
      R"({"data":[{"index":2,"embedding":[20,21]},{"index":0,"embedding":[0,1]},{"index":1,"embedding":[10,11]}]})"), result));
  ASSERT_NE(nullptr, result);
  ASSERT_EQ(3, result->element_count());
  auto *array = static_cast<ObJsonArray *>(result);
  for (int64_t i = 0; i < 3; ++i) {
    auto *vector = static_cast<ObJsonArray *>(array->get_value(i));
    EXPECT_EQ(i * 10, vector->get_value(0)->get_int());
  }
}

TEST_F(ProviderTest, openai_embeddings_reject_malformed_indices_and_vectors)
{
  const char *cases[] = {
    R"({})", R"({"data":{}})", R"({"data":[]})", R"({"data":[null]})",
    R"({"data":[{"embedding":[1]}]})",
    R"({"data":[{"index":-1,"embedding":[1]}]})",
    R"({"data":[{"index":1,"embedding":[1]}]})",
    R"({"data":[{"index":0.5,"embedding":[1]}]})",
    R"({"data":[{"index":"0","embedding":[1]}]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]})",
    R"({"data":[{"index":0,"embedding":null}]})",
    R"({"data":[{"index":0,"embedding":"abc"}]})",
    R"({"data":[{"index":0,"embedding":[]}]})",
    R"({"data":[{"index":0,"embedding":[true]}]})",
    R"({"data":[{"index":0,"embedding":["1"]}]})",
    R"({"data":[{"index":0,"embedding":[null]}]})",
    R"({"data":[{"index":0,"embedding":[1e100]}]})",
    R"({"data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[1,2]}]})"
  };
  ObOpenAIUtils::ObOpenAIEmbed provider;
  for (const char *json : cases) {
    SCOPED_TRACE(json);
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, parse(json), result));
    EXPECT_EQ(nullptr, result);
  }
}

TEST_F(ProviderTest, embeddings_reject_non_finite_numbers)
{
  ObJsonObject *response = parse(R"({"data":[{"index":0,"embedding":[1.0]}]})");
  auto *data = static_cast<ObJsonArray *>(response->get_value("data"));
  auto *item = static_cast<ObJsonObject *>(data->get_value(0));
  auto *vector = static_cast<ObJsonArray *>(item->get_value("embedding"));
  auto *value = static_cast<ObJsonDouble *>(vector->get_value(0));
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObIJsonBase *result = nullptr;
  value->set_value(std::numeric_limits<double>::infinity());
  EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, response, result));
  value->set_value(std::numeric_limits<double>::quiet_NaN());
  EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, response, result));
}

TEST_F(ProviderTest, openai_completion_requires_nonempty_string)
{
  const char *cases[] = {
    R"({})", R"({"choices":null})", R"({"choices":{}})", R"({"choices":[]})",
    R"({"choices":[null]})", R"({"choices":[{"message":[]}]})",
    R"({"choices":[{"message":{"content":null}}]})",
    R"({"choices":[{"message":{"content":[]}}]})",
    R"({"choices":[{"message":{"content":""}}]})"
  };
  ObOpenAIUtils::ObOpenAIComplete provider;
  for (const char *json : cases) {
    SCOPED_TRACE(json);
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, parse(json), result));
    EXPECT_EQ(nullptr, result);
  }
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(
      R"({"choices":[{"message":{"content":"hello"}}]})"), result));
  EXPECT_EQ(5, result->get_data_length());
}

TEST_F(ProviderTest, openai_requests_preserve_config_and_long_unicode_prompt)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  std::string long_text(100000, 'x');
  long_text += "中文\n\"";
  ObString content(long_text.size(), long_text.data());
  ObString model("test-model");
  ObString system;
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.get_body(allocator_, model, system, content,
      parse(R"({"temperature":0.25,"max_tokens":128})"), body));
  ASSERT_NE(nullptr, body->get_value("temperature"));
  EXPECT_EQ(128, body->get_value("max_tokens")->get_int());
  auto *messages = static_cast<ObJsonArray *>(body->get_value("messages"));
  ASSERT_EQ(1, messages->element_count());
  auto *message = static_cast<ObJsonObject *>(messages->get_value(0));
  auto *text = message->get_value("content");
  EXPECT_EQ(long_text, std::string(text->get_data(), text->get_data_length()));
}

TEST_F(ProviderTest, ollama_uses_native_nonstreaming_generation)
{
  ObOllamaUtils::ObOllamaComplete provider;
  ObString model("test-model"), system("system prompt"), content("user prompt");
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.get_body(allocator_, model, system, content,
      parse(R"({"stream":true,"temperature":0.25,"max_tokens":48,"options":{"seed":7}})"), body));
  EXPECT_EQ(nullptr, body->get_value("messages"));
  EXPECT_EQ(nullptr, body->get_value("max_tokens"));
  ASSERT_NE(nullptr, body->get_value("prompt"));
  EXPECT_FALSE(static_cast<ObJsonBoolean *>(body->get_value("stream"))->value());
  auto *options = static_cast<ObJsonObject *>(body->get_value("options"));
  ASSERT_NE(nullptr, options);
  EXPECT_EQ(48, options->get_value("num_predict")->get_int());
  EXPECT_EQ(7, options->get_value("seed")->get_int());
  EXPECT_DOUBLE_EQ(0.25, options->get_value("temperature")->get_double());
}

TEST_F(ProviderTest, ollama_completion_validates_response)
{
  ObOllamaUtils::ObOllamaComplete provider;
  const char *bad[] = {R"({})", R"({"response":null})", R"({"response":[]})",
                      R"({"response":""})", R"({"response":"partial","done":false})"};
  for (const char *json : bad) {
    ObIJsonBase *result = nullptr;
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, parse(json), result));
  }
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(
      R"({"response":"answer","done":true})"), result));
}

TEST_F(ProviderTest, ollama_embedding_supports_multiple_inputs_and_dimensions)
{
  ObOllamaUtils::ObOllamaEmbed provider;
  ObString model("embedding-model");
  ObArray<ObString> inputs;
  ASSERT_EQ(OB_SUCCESS, inputs.push_back(ObString("first")));
  ASSERT_EQ(OB_SUCCESS, inputs.push_back(ObString("second")));
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.get_body(allocator_, model, inputs, parse(R"({"dimensions":2})"), body));
  EXPECT_EQ(2, body->get_value("input")->element_count());
  EXPECT_EQ(2, body->get_value("dimensions")->get_int());
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(R"({"embeddings":[[1,2],[3,4]]})"), result));
  EXPECT_EQ(2, result->element_count());
  const char *bad[] = {R"({"embeddings":{}})", R"({"embeddings":[]})", R"({"embeddings":[null]})",
                      R"({"embeddings":[[1],[2,3]]})", R"({"embeddings":[["x"]]})"};
  for (const char *json : bad) {
    EXPECT_NE(OB_SUCCESS, provider.parse_output(allocator_, parse(json), result));
  }
}

TEST_F(ProviderTest, ollama_provider_is_case_insensitive_and_keyless)
{
  ObAIFuncIComplete *complete = nullptr;
  ObAIFuncIEmbed *embed = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::get_complete_provider(allocator_, ObString("oLlAmA"), complete));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::get_embed_provider(allocator_, ObString("ollama"), embed));
  ObString empty_key;
  ObArray<ObString> headers;
  ASSERT_EQ(OB_SUCCESS, complete->get_header(allocator_, empty_key, headers));
  ASSERT_EQ(1, headers.count());
  EXPECT_EQ(ObString("Content-Type: application/json"), headers.at(0));
  share::ObAiModelEndpointInfo endpoint;
  ASSERT_EQ(OB_SUCCESS, endpoint.parse_from_json_base(allocator_, ObString("endpoint"), *parse(
      R"({"ai_model_name":"model","url":"http://127.0.0.1:8000/api/generate","provider":"oLlAmA"})")));
  EXPECT_EQ(OB_SUCCESS, endpoint.merge_delta_endpoint(allocator_, *parse(R"({"access_key":""})")));
  EXPECT_NE(OB_SUCCESS, endpoint.merge_delta_endpoint(allocator_, *parse(R"({"provider":"OPENAI"})")));
}

TEST_F(ProviderTest, rerank_parsers_reject_wrong_containers)
{
  ObDashscopeUtils::ObDashscopeRerank dashscope;
  ObSiliconflowUtils::ObSiliconflowRerank siliconflow;
  ObIJsonBase *result = nullptr;
  EXPECT_NE(OB_SUCCESS, dashscope.parse_output(allocator_, parse(R"({"output":[]})"), result));
  EXPECT_NE(OB_SUCCESS, dashscope.parse_output(allocator_, parse(R"({"output":{"results":1}})"), result));
  EXPECT_NE(OB_SUCCESS, siliconflow.parse_output(allocator_, parse(R"({"results":{}})"), result));
}
TEST_F(ProviderTest, dashscope_preserves_config_and_restores_text_indices)
{
  ObDashscopeUtils::ObDashscopeEmbed provider;
  ObString model("model");
  ObArray<ObString> input;
  ASSERT_EQ(OB_SUCCESS, input.push_back(ObString("a")));
  ASSERT_EQ(OB_SUCCESS, input.push_back(ObString("b")));
  ObJsonObject *config = parse(R"({"dimensions":2})");
  ObJsonObject *body = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.get_body(allocator_, model, input, config, body));
  EXPECT_NE(nullptr, config->get_value("dimensions"));
  auto *parameters = static_cast<ObJsonObject *>(body->get_value("parameters"));
  EXPECT_EQ(2, parameters->get_value("dimension")->get_int());
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, parse(
      R"({"output":{"embeddings":[{"text_index":1,"embedding":[2,3]},{"text_index":0,"embedding":[0,1]}]}})"), result));
  auto *first = static_cast<ObJsonArray *>(static_cast<ObJsonArray *>(result)->get_value(0));
  EXPECT_EQ(0, first->get_value(0)->get_int());
}

TEST_F(ProviderTest, model_rejects_invalid_dimension_before_request)
{
  ObAIFuncExprInfo info(allocator_, T_FUN_SYS_AI_EMBED);
  info.type_ = share::EndpointType::DENSE_EMBEDDING;
  share::ObAiModelEndpointInfo endpoint;
  ObAIFuncModel model(allocator_, info, endpoint);
  ObArray<ObString> inputs;
  ObArray<ObString> results;
  ASSERT_EQ(OB_SUCCESS, inputs.push_back(ObString("text")));
  const char *bad[] = {R"({"dimensions":0})", R"({"dimensions":-1})", R"({"dimensions":1.5})",
                      R"({"dimensions":null})", R"({"dimensions":"2"})"};
  for (const char *json : bad) {
    EXPECT_EQ(OB_INVALID_ARGUMENT, model.call_dense_embedding_vector_v2(inputs, parse(json), results));
    EXPECT_TRUE(results.empty());
  }
}

} // namespace
} // namespace common
} // namespace oceanbase
