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

#define USING_LOG_PREFIX SQL_ENG
#include "ob_ai_func_utils.h"
#include <cmath>
#include "ob_ai_func_client.h"
#include "lib/utility/ob_sort.h"
#include "query/engine/expr/ob_ai_model_resolver.h"

namespace oceanbase 
{
namespace common 
{

namespace {
// Validate before casting: endpoint responses are untrusted JSON.
ObJsonNode *ai_object_member(ObIJsonBase *node, const char *key)
{
  return node != nullptr && node->json_type() == ObJsonNodeType::J_OBJECT
      ? static_cast<ObJsonObject *>(node)->get_value(key) : nullptr;
}

int validate_embedding(ObIAllocator &allocator, ObIJsonBase *embedding, int64_t &dimension)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(embedding) || embedding->json_type() != ObJsonNodeType::J_ARRAY
      || embedding->element_count() == 0) {
    ret = OB_INVALID_DATA;
  } else if (dimension > 0 && embedding->element_count() != dimension) {
    ret = OB_INVALID_DATA;
  } else {
    dimension = embedding->element_count();
    share::ObJsonReaderHelper reader(allocator);
    for (int64_t i = 0; OB_SUCC(ret) && i < dimension; ++i) {
      ObIJsonBase *value = nullptr;
      float number = 0;
      if (OB_FAIL(reader.get_array_element(embedding, i, value))) {
      } else if (!share::ObJsonHelper::is_number_type(value)) {
        ret = OB_INVALID_DATA;
      } else if (OB_FAIL(reader.get_float_value(value, number))) {
      } else if (!std::isfinite(number)) {
        ret = OB_INVALID_DATA;
      }
    }
  }
  return ret;
}

int parse_indexed_embeddings(ObIAllocator &allocator, ObJsonNode *data,
                             const char *index_key, ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  result = nullptr;
  ObJsonArray *ordered = nullptr;
  ObArray<ObJsonNode *> embeddings;
  int64_t dimension = 0;
  if (OB_ISNULL(data) || data->json_type() != ObJsonNodeType::J_ARRAY
      || data->element_count() == 0) {
    ret = OB_INVALID_DATA;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, ordered))) {
  } else {
    const int64_t count = data->element_count();
    for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
      if (OB_FAIL(embeddings.push_back(nullptr))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
      ObJsonNode *item = static_cast<ObJsonArray *>(data)->get_value(i);
      ObJsonNode *index = ai_object_member(item, index_key);
      ObJsonNode *embedding = ai_object_member(item, "embedding");
      int64_t position = -1;
      if (OB_ISNULL(index)) {
        ret = OB_INVALID_DATA;
      } else if (index->json_type() == ObJsonNodeType::J_INT) {
        position = index->get_int();
      } else if (index->json_type() == ObJsonNodeType::J_UINT && index->get_uint() < count) {
        position = index->get_uint();
      } else {
        ret = OB_INVALID_DATA;
      }
      if (OB_FAIL(ret)) {
      } else if (position < 0 || position >= count || embeddings.at(position) != nullptr) {
        ret = OB_INVALID_DATA;
      } else if (OB_FAIL(validate_embedding(allocator, embedding, dimension))) {
      } else {
        embeddings.at(position) = embedding;
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
      if (OB_ISNULL(embeddings.at(i))) {
        ret = OB_INVALID_DATA;
      } else if (OB_FAIL(ordered->append(embeddings.at(i)))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    result = ordered;
  }
  return ret;
}
} // namespace

int ObOpenAIUtils::get_header(common::ObIAllocator &allocator,
                              ObString &api_key,
                              common::ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  if (api_key.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    //["Authorization: Bearer %.*s", "Content-Type: application/json"]
    int auth_header_len = 1024;
    char *auth_header_str = (char *)allocator.alloc(auth_header_len);
    ObString content_type_str("Content-Type: application/json");
    ObString content_type_c_str;
    if (OB_FAIL(ob_write_string(allocator, content_type_str, content_type_c_str, true))) {
    } else if (OB_ISNULL(auth_header_str)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("Failed to allocate memory for auth header string and content type string", K(ret));
    } else {
      int auth_header_pos = snprintf(auth_header_str, auth_header_len,
                         "Authorization: Bearer %.*s", api_key.length(), api_key.ptr());
      if (auth_header_pos < 0) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(headers.push_back(ObString(auth_header_str)))) {
      } else if (OB_FAIL(headers.push_back(content_type_c_str))) {
      }
    }
  }
  return ret;
}
int ObOpenAIUtils::ObOpenAIComplete::get_header(common::ObIAllocator &allocator,
                                                ObString &api_key,
                                                common::ObArray<ObString> &headers) 
{
  return ObOpenAIUtils::get_header(allocator, api_key, headers);
}

int ObOpenAIUtils::ObOpenAIComplete::get_body(common::ObIAllocator &allocator,
                                              common::ObString &model,
                                              common::ObString &prompt,
                                              common::ObString &content,
                                              common::ObJsonObject *config,
                                              common::ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  if (model.empty() || content.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // {"model": "*", "messages": [{"role": "system", "content": "*"}, {"role": "user", "content": "*"}]}
    ObJsonObject *body_obj = nullptr;
    ObJsonString *model_str = nullptr;
    ObJsonArray *messages_array = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, body_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_str))) {
    } else if (OB_FAIL(body_obj->add("model", model_str))) {
    } else if (OB_FAIL(construct_messages_array(allocator, prompt, content, messages_array))) {
    } else if (OB_FAIL(body_obj->add("messages", messages_array))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::compact_json_object(allocator, config, body_obj))) {
    } else {
      body = body_obj;
    }
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIComplete::construct_messages_array(ObIAllocator &allocator, ObString &prompt, ObString &content, ObJsonArray *&messages)
{
  int ret = OB_SUCCESS;
  if (content.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    //messages: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}]
    ObJsonArray *messages_array = nullptr;
    ObJsonObject *sys_message_obj = nullptr;
    ObJsonObject *user_message_obj = nullptr;
    ObString system_str("system");
    ObString user_str("user");
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, messages_array))) {
    } else if(!prompt.empty()) {
      if (OB_FAIL(construct_message_obj(allocator, system_str, prompt, sys_message_obj))) {
      } else if (OB_FAIL(messages_array->append(sys_message_obj))) {
      }
    } 
    if (OB_SUCC(ret)) {
      if (OB_FAIL(construct_message_obj(allocator, user_str, content, user_message_obj))) {
      } else if (OB_FAIL(messages_array->append(user_message_obj))) {
      } else {
        messages = messages_array;
      }
    }
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIComplete::construct_message_obj(ObIAllocator &allocator, ObString &role, ObString &content, ObJsonObject *&message)
{
  int ret = OB_SUCCESS;
  if (role.empty() || content.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *message_obj = nullptr;
    ObJsonString *role_json_str = nullptr;
    ObJsonString *content_json_str = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, message_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, role, role_json_str))) {
    } else if (OB_FAIL(message_obj->add("role", role_json_str))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, content, content_json_str))) {
    } else if (OB_FAIL(message_obj->add("content", content_json_str))) {
    } else {
      message = message_obj;
    }
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIComplete::set_config_json_format(common::ObIAllocator &allocator, common::ObJsonObject *config)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(config)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // {"response_format":{"type":"json_object"}}
    ObString json_str("json_object");
    ObJsonString *json_str_obj = nullptr;
    ObJsonObject *json_obj = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, json_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, json_str, json_str_obj))) {
    } else if (OB_FAIL(json_obj->add("type", json_str_obj))) {
    } else if (OB_FAIL(config->add("response_format", json_obj))) {
    }
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIComplete::parse_output(ObIAllocator &allocator,
                                                  ObJsonObject *http_response,
                                                  ObIJsonBase *&result)
{
  UNUSED(allocator);
  int ret = OB_SUCCESS;
  result = nullptr;
  ObJsonNode *choices = ai_object_member(http_response, "choices");
  if (OB_ISNULL(choices) || choices->json_type() != ObJsonNodeType::J_ARRAY
      || choices->element_count() == 0) {
    ret = OB_INVALID_DATA;
  } else {
    ObJsonNode *message = ai_object_member(static_cast<ObJsonArray *>(choices)->get_value(0), "message");
    ObJsonNode *content = ai_object_member(message, "content");
    if (OB_ISNULL(content) || content->json_type() != ObJsonNodeType::J_STRING
        || content->get_data_length() == 0) {
      ret = OB_INVALID_DATA;
    } else {
      result = content;
    }
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIEmbed::get_header(common::ObIAllocator &allocator,
                                             ObString &api_key,
                                             common::ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObOpenAIUtils::get_header(allocator, api_key, headers))) {
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIEmbed::get_body(common::ObIAllocator &allocator,
                                           common::ObString &model,
                                           common::ObArray<ObString> &contents,
                                           common::ObJsonObject *config,
                                           common::ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  if (model.empty() || contents.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *body_obj = nullptr;
    ObJsonString *model_str = nullptr;
    ObJsonArray *input_array = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, body_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_str))) {
    } else if (OB_FAIL(body_obj->add("model", model_str))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::transform_array_to_json_array(allocator, contents, input_array))) {
    } else if (OB_FAIL(body_obj->add("input", input_array))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::compact_json_object(allocator, config, body_obj))) {
    } else {
      body = body_obj;
    }
  }
  return ret;
}

int ObOpenAIUtils::ObOpenAIEmbed::parse_output(ObIAllocator &allocator,
                                               ObJsonObject *http_response,
                                               ObIJsonBase *&result)
{
  return parse_indexed_embeddings(allocator, ai_object_member(http_response, "data"), "index", result);
}

int ObOllamaUtils::get_header(ObIAllocator &allocator, ObArray<ObString> &headers)
{
  int ret = OB_SUCCESS;
  ObString header("Content-Type: application/json");
  ObString copy;
  if (OB_FAIL(ob_write_string(allocator, header, copy, true))) {
  } else if (OB_FAIL(headers.push_back(copy))) {
  }
  return ret;
}

int ObOllamaUtils::ObOllamaComplete::get_header(common::ObIAllocator &allocator,
                                                common::ObString &api_key,
                                                common::ObArray<ObString> &headers) 
{
  return ObOllamaUtils::get_header(allocator, headers);
}

int ObOllamaUtils::ObOllamaComplete::get_body(ObIAllocator &allocator,
                                              ObString &model, ObString &prompt,
                                              ObString &content, ObJsonObject *config,
                                              ObJsonObject *&body)
{
  int ret = OB_SUCCESS;
  body = nullptr;
  ObJsonObject *request = nullptr;
  ObJsonObject *options = nullptr;
  ObJsonString *model_node = nullptr;
  ObJsonString *content_node = nullptr;
  ObJsonString *system_node = nullptr;
  ObJsonBoolean *stream = nullptr;
  if (model.empty() || content.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, request))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, options))) {
  } else if (OB_NOT_NULL(config)) {
    for (int64_t i = 0; OB_SUCC(ret) && i < config->element_count(); ++i) {
      ObString key;
      ObJsonNode *value = config->get_value(i);
      if (OB_FAIL(config->get_key(i, key))) {
      } else if (key == "options") {
        if (OB_ISNULL(value) || value->json_type() != ObJsonNodeType::J_OBJECT) {
          ret = OB_INVALID_ARGUMENT;
        } else if (OB_FAIL(ObAIFuncJsonUtils::compact_json_object(
                       allocator, static_cast<ObJsonObject *>(value), options))) {
        }
      } else if (key == "temperature" || key == "max_tokens") {
        if (OB_FAIL(options->add(key == "max_tokens" ? ObString("num_predict") : key, value))) {
        }
      } else if (key != "model" && key != "prompt" && key != "stream") {
        if (OB_FAIL(request->add(key, value))) {
        }
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_node))) {
  } else if (OB_FAIL(request->add("model", model_node))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, content, content_node))) {
  } else if (OB_FAIL(request->add("prompt", content_node))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_boolean(allocator, false, stream))) {
  } else if (OB_FAIL(request->add("stream", stream))) {
  } else if (options->element_count() > 0 && OB_FAIL(request->add("options", options))) {
  } else if (!prompt.empty()) {
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, prompt, system_node))) {
    } else if (OB_FAIL(request->add("system", system_node))) {
    }
  }
  if (OB_SUCC(ret)) {
    body = request;
  }
  return ret;
}

int ObOllamaUtils::ObOllamaComplete::set_config_json_format(common::ObIAllocator &allocator, common::ObJsonObject *config)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(config)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // {"format": "json"}
    ObString json_str("json");
    ObJsonString *json_str_obj = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, json_str, json_str_obj))) {
    } else if (OB_FAIL(config->add("format", json_str_obj))) {
    } 
  }
  return ret;
}

int ObOllamaUtils::ObOllamaComplete::parse_output(ObIAllocator &allocator,
                                                  ObJsonObject *http_response,
                                                  ObIJsonBase *&result)
{
  UNUSED(allocator);
  int ret = OB_SUCCESS;
  result = nullptr;
  ObJsonNode *response = ai_object_member(http_response, "response");
  ObJsonNode *done = ai_object_member(http_response, "done");
  if (OB_ISNULL(response) || response->json_type() != ObJsonNodeType::J_STRING
      || response->get_data_length() == 0 || OB_NOT_NULL(ai_object_member(http_response, "error"))) {
    ret = OB_INVALID_DATA;
  } else if (OB_NOT_NULL(done) && (done->json_type() != ObJsonNodeType::J_BOOLEAN
                                  || !static_cast<ObJsonBoolean *>(done)->value())) {
    ret = OB_INVALID_DATA;
  } else {
    result = response;
  }
  return ret;
}

int ObOllamaUtils::ObOllamaEmbed::get_header(common::ObIAllocator &allocator,
                                             ObString &api_key,
                                             common::ObArray<ObString> &headers) 
{
  return ObOllamaUtils::get_header(allocator, headers);
}

int ObOllamaUtils::ObOllamaEmbed::get_body(common::ObIAllocator &allocator,
                                           common::ObString &model,
                                           common::ObArray<ObString> &contents,
                                           common::ObJsonObject *config,
                                           common::ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  if (model.empty() || contents.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *body_obj = nullptr;
    ObJsonString *model_str = nullptr;
    ObJsonArray *input_array = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, body_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_str))) {
    } else if (OB_FAIL(body_obj->add("model", model_str))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::transform_array_to_json_array(allocator, contents, input_array))) {
    } else if (OB_FAIL(body_obj->add("input", input_array))) {
    } else {
      body = body_obj;
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(ObAIFuncJsonUtils::compact_json_object(allocator, config, body))) {
      }
    }
  }
  return ret;
}

int ObOllamaUtils::ObOllamaEmbed::parse_output(ObIAllocator &allocator,
                                               ObJsonObject *http_response,
                                               ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  result = nullptr;
  ObJsonNode *embeddings = ai_object_member(http_response, "embeddings");
  int64_t dimension = 0;
  if (OB_ISNULL(embeddings) || embeddings->json_type() != ObJsonNodeType::J_ARRAY
      || embeddings->element_count() == 0) {
    ret = OB_INVALID_DATA;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < embeddings->element_count(); ++i) {
      if (OB_FAIL(validate_embedding(allocator, static_cast<ObJsonArray *>(embeddings)->get_value(i), dimension))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    result = embeddings;
  }
  return ret;
}

int ObDashscopeUtils::get_header(common::ObIAllocator &allocator,
                                 common::ObString &api_key,
                                 common::ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObOpenAIUtils::get_header(allocator, api_key, headers))) {
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeComplete::get_header(common::ObIAllocator &allocator, 
                                                      common::ObString &api_key,
                                                      common::ObArray<ObString> &headers) 
{
  return ObOpenAIUtils::get_header(allocator, api_key, headers);
}

int ObDashscopeUtils::ObDashscopeComplete::get_body(common::ObIAllocator &allocator,
                                                    common::ObString &model,
                                                    common::ObString &prompt,
                                                    common::ObString &content,
                                                    common::ObJsonObject *config,
                                                    common::ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(model) || OB_ISNULL(content)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // {"model": "*", "input": {"messages": [{"role": "system", "content": "*"}, {"role": "user", "content": "*"}]}, "parameters": {}}
    ObJsonObject *body_obj = nullptr;
    ObJsonString *model_str = nullptr;
    ObJsonObject *input_obj = nullptr;

    if (OB_ISNULL(config)) {
      if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config))) {
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, body_obj))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_str))) {
      } else if (OB_FAIL(body_obj->add("model", model_str))) {
      } else if (OB_FAIL(ObDashscopeUtils::ObDashscopeComplete::construct_input_obj(allocator, prompt, content, input_obj))) {
      } else if (OB_FAIL(body_obj->add("input", input_obj))) {
      } else if (OB_FAIL(ObDashscopeUtils::ObDashscopeComplete::set_config_result_format(allocator, config))) {
      } else if (OB_FAIL(body_obj->add("parameters", config))) {
      } else {
        body = body_obj;
      }
    }
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeComplete::set_config_result_format(ObIAllocator &allocator, ObJsonObject *config)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(config)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_NOT_NULL(config->get_value("result_format"))) {
    if (OB_FAIL(config->remove("result_format"))) {
    }
  } 
  if (OB_SUCC(ret)) {
    // {"result_format": "message"}
    ObJsonString *result_format_str = nullptr;
    ObString result_format_str_val("message");
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, result_format_str_val, result_format_str))) {
    } else if (OB_FAIL(config->add("result_format", result_format_str))) {
    }
  } 
  return ret;
}

int ObDashscopeUtils::ObDashscopeComplete::construct_input_obj(ObIAllocator &allocator, ObString &prompt, ObString &content, ObJsonObject *&input_obj)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(content)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *obj = nullptr;
    ObJsonArray *messages_array = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, obj))) {
    } else if (OB_FAIL(ObOpenAIUtils::ObOpenAIComplete::construct_messages_array(allocator, prompt, content, messages_array))) {
    } else if (OB_FAIL(obj->add("messages", messages_array))) {
    } else {
      input_obj = obj;
    }
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeComplete::set_config_json_format(ObIAllocator &allocator, ObJsonObject *config)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(config)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    //{"type": "json_object"}
    ObJsonString *type_str = nullptr;
    ObString type_str_val("json_object");
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, type_str_val, type_str))) {
    } else if (OB_FAIL(config->add("type", type_str))) {
    }
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeComplete::parse_output(ObIAllocator &allocator,
                                                       ObJsonObject *http_response,
                                                       ObIJsonBase *&result)
{
  ObJsonNode *output = ai_object_member(http_response, "output");
  result = nullptr;
  if (OB_ISNULL(output) || output->json_type() != ObJsonNodeType::J_OBJECT) {
    return OB_INVALID_DATA;
  }
  ObOpenAIUtils::ObOpenAIComplete parser;
  return parser.parse_output(allocator, static_cast<ObJsonObject *>(output), result);
}

int ObDashscopeUtils::ObDashscopeEmbed::get_header(common::ObIAllocator &allocator,
                                                   common::ObString &api_key,
                                                   common::ObArray<ObString> &headers) 
{
  return ObDashscopeUtils::get_header(allocator, api_key, headers);
}

int ObDashscopeUtils::ObDashscopeEmbed::get_body(ObIAllocator &allocator,
                                                 ObString &model, ObArray<ObString> &contents,
                                                 ObJsonObject *config, ObJsonObject *&body)
{
  int ret = OB_SUCCESS;
  body = nullptr;
  ObJsonObject *request = nullptr;
  ObJsonObject *input = nullptr;
  ObJsonObject *parameters = nullptr;
  ObJsonString *model_node = nullptr;
  ObJsonArray *texts = nullptr;
  if (model.empty() || contents.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, request))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_node))) {
  } else if (OB_FAIL(request->add("model", model_node))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, input))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::transform_array_to_json_array(allocator, contents, texts))) {
  } else if (OB_FAIL(input->add("texts", texts))) {
  } else if (OB_FAIL(request->add("input", input))) {
  } else if (OB_NOT_NULL(config) && config->element_count() > 0) {
    if (OB_ISNULL(parameters = static_cast<ObJsonObject *>(config->clone(&allocator, true)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_NOT_NULL(parameters->get_value("dimensions"))
               && OB_FAIL(parameters->rename_key("dimensions", "dimension"))) {
    } else if (OB_FAIL(request->add("parameters", parameters))) {
    }
  }
  if (OB_SUCC(ret)) {
    body = request;
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeEmbed::parse_output(ObIAllocator &allocator,
                                                    ObJsonObject *http_response,
                                                    ObIJsonBase *&result)
{
  ObJsonNode *output = ai_object_member(http_response, "output");
  return parse_indexed_embeddings(allocator, ai_object_member(output, "embeddings"), "text_index", result);
}

int ObDashscopeUtils::ObDashscopeRerank::get_header(common::ObIAllocator &allocator,
                                                    common::ObString &api_key,
                                                    common::ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObOpenAIUtils::get_header(allocator, api_key, headers))) {
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeRerank::get_body(common::ObIAllocator &allocator,
                                                  common::ObString &model,
                                                  common::ObString &query,
                                                  common::ObJsonArray *document_array,
                                                  common::ObJsonObject *config,
                                                  common::ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(model) || OB_ISNULL(query) || OB_ISNULL(document_array)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *body_obj = nullptr;
    ObJsonString *model_str = nullptr;
    ObJsonObject *input_obj = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, body_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_str))) {
    } else if (OB_FAIL(body_obj->add("model", model_str))) {
    } else if (OB_FAIL(ObDashscopeUtils::ObDashscopeRerank::get_input_obj(allocator, query, document_array, input_obj))) {
    } else if (OB_FAIL(body_obj->add("input", input_obj))) {
    } else if (config != nullptr && config->element_count() > 0) {
      if (OB_FAIL(body_obj->add("parameters", config))) {
      }
    } 
    if (OB_SUCC(ret)) {
      body = body_obj;
    }
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeRerank::get_input_obj(common::ObIAllocator &allocator,
                                                      common::ObString &query,
                                                      common::ObJsonArray *document_array,
                                                      common::ObJsonObject *&input_obj) 
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(query) || OB_ISNULL(document_array)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *obj = nullptr;
    ObJsonString *query_str = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, query, query_str))) {
    } else if (OB_FAIL(obj->add("query", query_str))) {
    } else if (OB_FAIL(obj->add("documents", document_array))) {
    } else {
      input_obj = obj;
    }
  }
  return ret;
}

int ObDashscopeUtils::ObDashscopeRerank::parse_output(ObIAllocator &allocator,
                                                      ObJsonObject *http_response,
                                                      ObIJsonBase *&result)
{
  UNUSED(allocator);
  result = nullptr;
  ObJsonNode *output = ai_object_member(http_response, "output");
  ObJsonNode *results = ai_object_member(output, "results");
  if (OB_ISNULL(results) || results->json_type() != ObJsonNodeType::J_ARRAY) {
    return OB_INVALID_DATA;
  }
  result = results;
  return OB_SUCCESS;
}

int ObSiliconflowUtils::get_header(common::ObIAllocator &allocator,
                                 common::ObString &api_key,
                                 common::ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObOpenAIUtils::get_header(allocator, api_key, headers))) {
  }
  return ret;
}

int ObSiliconflowUtils::ObSiliconflowRerank::get_header(common::ObIAllocator &allocator,
                                                    common::ObString &api_key,
                                                    common::ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObOpenAIUtils::get_header(allocator, api_key, headers))) {
  }
  return ret;
}

int ObSiliconflowUtils::ObSiliconflowRerank::get_body(common::ObIAllocator &allocator,
                                                  common::ObString &model,
                                                  common::ObString &query,
                                                  common::ObJsonArray *document_array,
                                                  common::ObJsonObject *config,
                                                  common::ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(model) || OB_ISNULL(query) || OB_ISNULL(document_array)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonObject *body_obj = nullptr;
    ObJsonString *model_str = nullptr;
    ObJsonString *query_str = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, body_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, model, model_str))) {
    } else if (OB_FAIL(body_obj->add("model", model_str))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, query, query_str))) {
    } else if (OB_FAIL(body_obj->add("query", query_str))) {
    } else if (OB_FAIL(body_obj->add("documents", document_array))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::compact_json_object(allocator, config, body_obj))) {
    } else {
      body = body_obj;
    }
  }
  return ret;
}

int ObSiliconflowUtils::ObSiliconflowRerank::parse_output(ObIAllocator &allocator,
                                                       ObJsonObject *http_response,
                                                       ObIJsonBase *&result)
{
  UNUSED(allocator);
  result = nullptr;
  ObJsonNode *results = ai_object_member(http_response, "results");
  if (OB_ISNULL(results) || results->json_type() != ObJsonNodeType::J_ARRAY) {
    return OB_INVALID_DATA;
  }
  result = results;
  return OB_SUCCESS;
}

int ObAIFuncUtils::get_header(ObIAllocator &allocator, 
                              const ObAIFuncExprInfo &info,
                              const share::ObAiModelEndpointInfo &endpoint_info,
                              ObArray<ObString> &headers) 
{
  int ret = OB_SUCCESS;
  ObString unencrypted_access_key;
  if (OB_FAIL(endpoint_info.get_unencrypted_access_key(allocator, unencrypted_access_key))) {
  } else if (ObAIFuncUtils::is_completion_type(&info)) {
    ObAIFuncIComplete *complete_provider = nullptr;
    if (OB_FAIL(get_complete_provider(allocator, endpoint_info.get_provider(), complete_provider))) {
    } else if (OB_FAIL(complete_provider->get_header(allocator, unencrypted_access_key, headers))) {
    }
  } else if (ObAIFuncUtils::is_dense_embedding_type(&info)) {
    ObAIFuncIEmbed *embed_provider = nullptr;
    if (OB_FAIL(get_embed_provider(allocator, endpoint_info.get_provider(), embed_provider))) {
    } else if (OB_FAIL(embed_provider->get_header(allocator, unencrypted_access_key, headers))) {
    }
  } else if (ObAIFuncUtils::is_rerank_type(&info)) {
    ObAIFuncIRerank *rerank_provider = nullptr;
    if (OB_FAIL(get_rerank_provider(allocator, endpoint_info.get_provider(), rerank_provider))) {
    } else if (OB_FAIL(rerank_provider->get_header(allocator, unencrypted_access_key, headers))) {
    }
  }
  return ret;
}

int ObAIFuncUtils::get_complete_body(ObIAllocator &allocator, 
                                    const ObAIFuncExprInfo &info, 
                                    const share::ObAiModelEndpointInfo &endpoint_info,
                                    ObString &prompt, 
                                    ObString &content, 
                                    ObJsonObject *config,
                                    ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  ObString request_model_name = info.model_;
  if (!endpoint_info.get_request_model_name().empty()) {
    request_model_name = endpoint_info.get_request_model_name();
  }

  ObAIFuncIComplete *complete_provider = nullptr;
  if (OB_FAIL(get_complete_provider(allocator, endpoint_info.get_provider(), complete_provider))) {
  } else if (OB_FAIL(complete_provider->get_body(allocator, request_model_name, prompt, content, config, body))) {
  }
  return ret;
}

int ObAIFuncUtils::set_json_format_config(ObIAllocator &allocator, const ObString &provider, ObJsonObject *config)
{
  int ret = OB_SUCCESS;
  ObAIFuncIComplete *complete_provider = nullptr;
  if (OB_FAIL(get_complete_provider(allocator, provider, complete_provider))) {
  } else if (OB_FAIL(complete_provider->set_config_json_format(allocator, config))) {
  } 
  return ret;
}

int ObAIFuncUtils::get_embed_body(ObIAllocator &allocator, 
                                  const ObAIFuncExprInfo &info, 
                                  const share::ObAiModelEndpointInfo &endpoint_info,
                                  ObArray<ObString> &contents, 
                                  ObJsonObject *config,
                                  ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  ObString request_model_name = info.model_;
  if (!endpoint_info.get_request_model_name().empty()) {
    request_model_name = endpoint_info.get_request_model_name();
  }

  ObAIFuncIEmbed *embed_provider = nullptr;
  if (OB_FAIL(get_embed_provider(allocator, endpoint_info.get_provider(), embed_provider))) {
  } else if (OB_FAIL(embed_provider->get_body(allocator, request_model_name, contents, config, body))) {
  }
  return ret;
}

int ObAIFuncUtils::get_rerank_body(ObIAllocator &allocator, 
                                   const ObAIFuncExprInfo &info,
                                   const share::ObAiModelEndpointInfo &endpoint_info,
                                   ObString &query,
                                   ObJsonArray *document_array,
                                   ObJsonObject *config,
                                   ObJsonObject *&body) 
{
  int ret = OB_SUCCESS;
  ObString request_model_name = info.model_;
  if (!endpoint_info.get_request_model_name().empty()) {
    request_model_name = endpoint_info.get_request_model_name();
  }

  ObAIFuncIRerank *rerank_provider = nullptr;
  if (OB_FAIL(get_rerank_provider(allocator, endpoint_info.get_provider(), rerank_provider))) {
  } else if (OB_FAIL(rerank_provider->get_body(allocator, request_model_name, query, document_array, config, body))) {
  }
  return ret;
}

int ObAIFuncUtils::parse_complete_output(ObIAllocator &allocator, 
                                        const share::ObAiModelEndpointInfo &endpoint_info,
                                        ObJsonObject *http_response,
                                        ObIJsonBase *&result)                                                 
{
  int ret = OB_SUCCESS;
  ObAIFuncIComplete *complete_provider = nullptr;
  if (OB_FAIL(get_complete_provider(allocator, endpoint_info.get_provider(), complete_provider))) {
  } else if (OB_FAIL(complete_provider->parse_output(allocator, http_response, result))) {
  }
  return ret;
}

int ObAIFuncUtils::parse_embed_output(ObIAllocator &allocator, 
                                      const share::ObAiModelEndpointInfo &endpoint_info,
                                      ObJsonObject *http_response,
                                      ObIJsonBase *&result) 
{
  int ret = OB_SUCCESS;
  ObAIFuncIEmbed *embed_provider = nullptr;
  if (OB_FAIL(get_embed_provider(allocator, endpoint_info.get_provider(), embed_provider))) {
  } else if (OB_FAIL(embed_provider->parse_output(allocator, http_response, result))) {
  }
  return ret;
}

int ObAIFuncUtils::parse_rerank_output(ObIAllocator &allocator, 
                                       const share::ObAiModelEndpointInfo &endpoint_info,
                                       ObJsonObject *http_response,
                                       ObIJsonBase *&result) 
{
  int ret = OB_SUCCESS;
  ObAIFuncIRerank *rerank_provider = nullptr;
  if (OB_FAIL(get_rerank_provider(allocator, endpoint_info.get_provider(), rerank_provider))) {
  } else if (OB_FAIL(rerank_provider->parse_output(allocator, http_response, result))) {
  }
  return ret;
}

int ObAIFuncJsonUtils::get_json_object(ObIAllocator &allocator, ObJsonObject *&obj_node)
{
  int ret = OB_SUCCESS;
  ObJsonObject *j_obj = OB_NEWx(ObJsonObject, &allocator, &allocator);
  if (OB_ISNULL(j_obj)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    obj_node = j_obj;
  }
  return ret;
}

int ObAIFuncJsonUtils::get_json_array(ObIAllocator &allocator, ObJsonArray *&array_node)
{
  int ret = OB_SUCCESS;
  ObJsonArray *j_array = OB_NEWx(ObJsonArray, &allocator, &allocator);
  if (OB_ISNULL(j_array)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    array_node = j_array; 
  }
  return ret;
}

int ObAIFuncJsonUtils::get_json_string(ObIAllocator &allocator, ObString &str, ObJsonString *&str_node)
{
  int ret = OB_SUCCESS;
  ObJsonString *j_str = OB_NEWx(ObJsonString, &allocator,str);
  if (OB_ISNULL(j_str)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    str_node = j_str;
  }
  return ret;
}

int ObAIFuncJsonUtils::get_json_int(ObIAllocator &allocator, int64_t num, ObJsonInt *&int_node)
{
  int ret = OB_SUCCESS;
  ObJsonInt *j_int = OB_NEWx(ObJsonInt, &allocator, num);
  if (OB_ISNULL(j_int)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    int_node = j_int;
  }
  return ret;
}

int ObAIFuncJsonUtils::get_json_boolean(ObIAllocator &allocator, bool value, ObJsonBoolean *&boolean_node)
{
  int ret = OB_SUCCESS;
  ObJsonBoolean *j_bool = OB_NEWx(ObJsonBoolean, &allocator, value);
  if (OB_ISNULL(j_bool)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    boolean_node = j_bool;
  }
  return ret;
}

int ObAIFuncJsonUtils::print_json_to_str(ObIAllocator &allocator, ObIJsonBase *base_node, ObString &str)
{
  int ret = OB_SUCCESS;
  ObJsonBuffer buffer(&allocator);
  if (OB_ISNULL(base_node)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(base_node->print(buffer, 0))) {
  } else if (OB_FAIL(buffer.get_result_string(str))) {
  }
  return ret;
}

int ObAIFuncJsonUtils::get_json_object_form_str(ObIAllocator &allocator, ObString &str, ObJsonObject *&obj_node)
{
  int ret = OB_SUCCESS;
  obj_node = nullptr;
  ObIJsonBase *parsed = nullptr;
  if (OB_FAIL(ObJsonBaseFactory::get_json_base(&allocator, str, ObJsonInType::JSON_TREE,
                                              ObJsonInType::JSON_TREE, parsed))) {
  } else if (OB_ISNULL(parsed) || parsed->json_type() != ObJsonNodeType::J_OBJECT) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    obj_node = static_cast<ObJsonObject *>(parsed);
  }
  return ret;
}

int ObAIFuncJsonUtils::compact_json_object(ObIAllocator &allocator, ObJsonObject *obj_node, ObJsonObject *compact_obj)
{
  int ret = OB_SUCCESS;
  // add all members of obj_node to compact_obj
  if (OB_ISNULL(compact_obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(obj_node)) {
    // do nothing
  } else {
    ObString key;
    int64_t count = obj_node->element_count();
    for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
      ObJsonNode *j_node = obj_node->get_value(i);
      if (OB_ISNULL(j_node)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(obj_node->get_key(i, key))) {
      } else if (OB_FAIL(compact_obj->add(key, j_node))) {
      }
    }
  }
  return ret;
}

int ObAIFuncPromptUtils::replace_meta_prompt(ObIAllocator &allocator, ObString &meta_prompt,
    ObString &key, ObString &content, ObString &result)
{
  int ret = OB_SUCCESS;
  int64_t key_pos = -1;
  if (meta_prompt.empty() || key.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // ObString is length-delimited; STRSTR would read past its allocation.
    for (int64_t i = 0; key_pos < 0 && i <= meta_prompt.length() - key.length(); ++i) {
      if (MEMCMP(meta_prompt.ptr() + i, key.ptr(), key.length()) == 0) {
        key_pos = i;
      }
    }
    if (key_pos < 0) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      ObStringBuffer buffer(&allocator);
      const int64_t tail_pos = key_pos + key.length();
      if (OB_FAIL(buffer.append(meta_prompt.ptr(), key_pos))) {
      } else if (OB_FAIL(buffer.append(content))) {
      } else if (OB_FAIL(buffer.append(meta_prompt.ptr() + tail_pos, meta_prompt.length() - tail_pos))) {
      } else if (buffer.length() > INT32_MAX) {
        ret = OB_SIZE_OVERFLOW;
      } else {
        ret = buffer.get_result_string(result);
      }
    }
  }
  return ret;
}

int ObAIFuncJsonUtils::transform_array_to_json_array(ObIAllocator &allocator, ObArray<ObString> &contents, ObJsonArray *&array)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(array = OB_NEWx(ObJsonArray, &allocator, &allocator))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    for (int i = 0; OB_SUCC(ret) && i < contents.count(); i++) {
      ObString content = contents[i];
      ObJsonString *j_str = OB_NEWx(ObJsonString, &allocator, content);
      if (OB_ISNULL(j_str)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(array->append(j_str))) {
      }
    }
  }
  return ret;
}

int ObAIFuncUtils::get_complete_provider(ObIAllocator &allocator, const ObString &provider, ObAIFuncIComplete *&complete_provider)
{
  int ret = OB_SUCCESS;
  if (provider.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_function, provider is empty");
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::OPENAI) 
      || ob_provider_check(provider, ObAIFuncProviderUtils::ALIYUN) 
      || ob_provider_check(provider, ObAIFuncProviderUtils::DEEPSEEK)
      || ob_provider_check(provider, ObAIFuncProviderUtils::SILICONFLOW)
      || ob_provider_check(provider, ObAIFuncProviderUtils::HUNYUAN)) {
    complete_provider = OB_NEWx(ObOpenAIUtils::ObOpenAIComplete, &allocator);
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::OLLAMA)) {
    complete_provider = OB_NEWx(ObOllamaUtils::ObOllamaComplete, &allocator);
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::DASHSCOPE)) {
    complete_provider = OB_NEWx(ObDashscopeUtils::ObDashscopeComplete, &allocator);
  } else {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "this provider current is");
  } 
  if (OB_SUCC(ret) && OB_ISNULL(complete_provider)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  return ret;
}

int ObAIFuncUtils::get_embed_provider(ObIAllocator &allocator, const ObString &provider, ObAIFuncIEmbed *&embed_provider)
{
  int ret = OB_SUCCESS;
  if (provider.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_function, provider is empty");
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::OPENAI) 
      || ob_provider_check(provider, ObAIFuncProviderUtils::ALIYUN)
      || ob_provider_check(provider, ObAIFuncProviderUtils::HUNYUAN)
      || ob_provider_check(provider, ObAIFuncProviderUtils::SILICONFLOW)) {
    embed_provider = OB_NEWx(ObOpenAIUtils::ObOpenAIEmbed, &allocator);
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::OLLAMA)) {
    embed_provider = OB_NEWx(ObOllamaUtils::ObOllamaEmbed, &allocator);
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::DASHSCOPE)) {
    embed_provider = OB_NEWx(ObDashscopeUtils::ObDashscopeEmbed, &allocator);
  } else {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "this provider current is");
  } 
  if (OB_SUCC(ret) && OB_ISNULL(embed_provider)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  return ret;
}

int ObAIFuncUtils::get_rerank_provider(ObIAllocator &allocator, const ObString &provider, ObAIFuncIRerank *&rerank_provider)
{
  int ret = OB_SUCCESS;
  if (provider.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_function, provider is empty");
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::SILICONFLOW)
          || ob_provider_check(provider, ObAIFuncProviderUtils::COHERE)) {
    rerank_provider = OB_NEWx(ObSiliconflowUtils::ObSiliconflowRerank, &allocator);
  } else if (ob_provider_check(provider, ObAIFuncProviderUtils::DASHSCOPE)) {
    rerank_provider = OB_NEWx(ObDashscopeUtils::ObDashscopeRerank, &allocator);
  } else {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "rerank support siliconflow, cohere and ailiyun-dashscope, this provider current is");
  } 
  if (OB_SUCC(ret) && OB_ISNULL(rerank_provider)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  return ret;
}

int ObAIFuncUtils::check_info_type_completion(const ObAIFuncExprInfo *info)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(info)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_complete, info is null");
  } else if (!is_completion_type(info)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_complete, info type is not completion");
  }
  return ret;
}

int ObAIFuncUtils::check_info_type_dense_embedding(const ObAIFuncExprInfo *info)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(info)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, info is null");
  } else if (!is_dense_embedding_type(info)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, info type is not dense embedding");
  }
  return ret;
}

int ObAIFuncUtils::check_info_type_rerank(const ObAIFuncExprInfo *info)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(info)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, info is null");
  } else if (!is_rerank_type(info)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, info type is not rerank");
  }
  return ret;
}

int ObAIFuncUtils::set_string_result(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res, ObString &res_str)
{
  int ret = OB_SUCCESS;
  ObTextStringDatumResult text_result(expr.datum_meta_.type_, &expr, &ctx, &res);
  int64_t res_len = res_str.length();
  if (OB_FAIL(text_result.init(res_len))) {
  } else if (OB_FAIL(text_result.append(res_str))) {
  } else {
    text_result.set_result();
  }
  return ret;
}

int ObAIFuncUtils::get_ai_func_info(ObIAllocator &allocator, const ObString &model_id,
                                    share::schema::ObSchemaGetterGuard &guard,
                                    ObAIFuncExprInfo *&info)
{
  int ret = OB_SUCCESS;
  if (model_id.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_function, model_id is empty");
  } else {
    ObAIFuncExprInfo *info_obj = OB_NEWx(ObAIFuncExprInfo, (&allocator), allocator, T_FUN_SYS_AI_COMPLETE);
    if (OB_ISNULL(info_obj)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(info_obj->init(allocator, model_id, guard))) {
    } else {
      info = info_obj;
    }
  }
  return ret;
}

bool ObAIFuncUtils::is_provider_support_base64(const ObString &provider)
{
  return ob_provider_check(provider, ObAIFuncProviderUtils::OPENAI)
      || ob_provider_check(provider, ObAIFuncProviderUtils::SILICONFLOW);
}

int ObAIFuncUtils::decode_base64_embedding_array(const ObIJsonBase &embedding_jbase,
                                                 ObIAllocator &allocator,
                                                 const int64_t dimension,
                                                 float *&vector)
{
  int ret = OB_SUCCESS;
  if (embedding_jbase.json_type() != ObJsonNodeType::J_STRING) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    const char *encoded_embedding = embedding_jbase.get_data();
    uint64_t encoded_embedding_len = embedding_jbase.get_data_length();
    uint64_t decoded_buf_len = ObBase64Encoder::needed_decoded_length(encoded_embedding_len);
    uint8_t *decoded_buf = nullptr;
    int64_t pos = 0;
    if (decoded_buf_len <= 0) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_ISNULL(decoded_buf = static_cast<uint8_t *>(allocator.alloc(decoded_buf_len)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(ObBase64Encoder::decode(encoded_embedding, encoded_embedding_len,
                                               decoded_buf, decoded_buf_len, pos))) {
    } else if (pos != dimension * sizeof(float)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      vector = reinterpret_cast<float *>(decoded_buf);
    }
  }
  return ret;
}

int ObAIFuncUtils::decode_float_embedding_array(const ObIJsonBase &embedding_jbase,
                                                ObIAllocator &allocator,
                                                share::ObJsonReaderHelper &json_reader,
                                                const int64_t dimension,
                                                float *&vector)
{
  int ret = OB_SUCCESS;
  float *tmp_vector = nullptr;
  if (!share::ObJsonHelper::is_array_type(&embedding_jbase)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    uint64_t embedding_size = json_reader.get_array_size(&embedding_jbase);
    if (embedding_size != dimension) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      tmp_vector = static_cast<float*>(allocator.alloc(dimension * sizeof(float)));
      if (OB_ISNULL(tmp_vector)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        for (uint64_t i = 0; i < dimension && OB_SUCC(ret); i++) {
          ObIJsonBase *value = nullptr;
          if (OB_FAIL(json_reader.get_array_element(&embedding_jbase, i, value))) {
          } else if (!share::ObJsonHelper::is_number_type(value)) {
            ret = OB_INVALID_ARGUMENT;
          } else {
            float f_value = 0.0;
            if (OB_FAIL(json_reader.get_float_value(value, f_value))) {
            } else {
              tmp_vector[i] = f_value;
            }
          }
        }
      }
    }
  }

  if (OB_SUCC(ret)) {
    vector = tmp_vector;
  }
  return ret;
}

int ObAIFuncUtils::get_ai_func_info(ObIAllocator &allocator, const ObString &model_id, ObAIFuncExprInfo *&info)
{
  int ret = OB_SUCCESS;
  share::schema::ObMultiVersionSchemaService *schema_service = GCTX.schema_service_;
  share::schema::ObSchemaGetterGuard guard;
  
  if (OB_ISNULL(schema_service)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(schema_service->get_runtime_schema_guard(guard))) {
  } else if (OB_FAIL(get_ai_func_info(allocator, model_id, guard, info))) {
  }
  return ret;
}

int ObAIFuncModel::call_completion(ObString &prompt, ObJsonObject *config, ObString &result)
{
  int ret = OB_SUCCESS;
  ObArray<ObString> headers;
  ObJsonObject *body = nullptr;
  ObJsonObject *response = nullptr;
  ObIJsonBase *result_base = nullptr;
  ObAIFuncIComplete *complete_provider = nullptr;
  ObString prompt_str;
  ObString result_str;
  ObAIFuncClient client;
  ObString unencrypted_access_key;
  ObString request_model_name = get_request_model_name();
  if (!is_completion_type()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_complete, info type is not completion");
  } else if (OB_FAIL(ObAIFuncUtils::get_complete_provider(*allocator_, endpoint_info_.get_provider(), complete_provider))) {
  } else if (OB_FAIL(endpoint_info_.get_unencrypted_access_key(*allocator_, unencrypted_access_key))) {
  } else if (OB_FAIL(complete_provider->get_header(*allocator_, unencrypted_access_key, headers))) {
  } else if (OB_FAIL(complete_provider->get_body(*allocator_, request_model_name, prompt_str, prompt, config, body))) {
  } else if (OB_FAIL(client.send_post(*allocator_, endpoint_info_.get_url(), headers, body, response))) {
  } else if (OB_FAIL(complete_provider->parse_output(*allocator_, response, result_base))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::print_json_to_str(*allocator_, result_base, result_str))) {
  } else {
    result = result_str;
  }
  if (ret == OB_INVALID_DATA) {
    FORWARD_USER_ERROR(ret, "unexpected completion response");
  }
  return ret;
}

int ObAIFuncModel::call_completion_vector(ObArray<ObString> &prompts, ObJsonObject *config,
                                          ObArray<ObString> &results)
{
  int ret = OB_SUCCESS;
  results.reset();
  ObArray<ObString> headers;
  ObArray<ObString> validated_results;
  ObArray<ObJsonObject *> bodies;
  ObArray<ObJsonObject *> responses;
  ObAIFuncIComplete *provider = nullptr;
  ObString system_prompt;
  ObString access_key;
  ObString model = get_request_model_name();
  ObAIFuncClient client;
  if (!is_completion_type() || prompts.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncUtils::get_complete_provider(*allocator_, endpoint_info_.get_provider(), provider))) {
  } else if (OB_FAIL(endpoint_info_.get_unencrypted_access_key(*allocator_, access_key))) {
  } else if (OB_FAIL(provider->get_header(*allocator_, access_key, headers))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < prompts.count(); ++i) {
      ObJsonObject *body = nullptr;
      if (OB_FAIL(provider->get_body(*allocator_, model, system_prompt, prompts.at(i), config, body))) {
      } else if (OB_FAIL(bodies.push_back(body))) {
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(client.send_post_batch(*allocator_, endpoint_info_.get_url(), headers, bodies, responses))) {
  } else if (responses.count() != prompts.count()) {
    ret = OB_INVALID_DATA;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < responses.count(); ++i) {
      ObIJsonBase *parsed = nullptr;
      ObString encoded;
      if (OB_FAIL(provider->parse_output(*allocator_, responses.at(i), parsed))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::print_json_to_str(*allocator_, parsed, encoded))) {
      } else if (OB_FAIL(validated_results.push_back(encoded))) {
      }
    }
  }
  if (OB_SUCC(ret) && OB_FAIL(results.assign(validated_results))) {
    results.reset();
  }
  return ret;
}

int ObAIFuncModel::call_dense_embedding(ObString &content, ObJsonObject *config, ObString &result)
{
  int ret = OB_SUCCESS;
  ObArray<ObString> contents;
  ObArray<ObString> results;
  if (content.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, input is empty");
  } else if (!is_dense_embedding_type()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, info type is not dense embedding");
  } else if (OB_FAIL(contents.push_back(content))) {
  } else if (OB_FAIL(call_dense_embedding_vector_v2(contents, config, results))) {
  } else if (results.count() != 1) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    result = results[0];
  }
  return ret;
}

int ObAIFuncModel::call_dense_embedding_vector(ObArray<ObString> &contents,
                                                ObJsonObject *config,
                                                ObArray<ObString> &results)
{
  return call_dense_embedding_vector_v2(contents, config, results);
}

int ObAIFuncModel::call_dense_embedding_vector_v2(ObArray<ObString> &contents,
                                                   ObJsonObject *config,
                                                   ObArray<ObString> &results)
{
  int ret = OB_SUCCESS;
  results.reset();
  ObArray<ObString> headers;
  ObArray<ObString> validated_results;
  ObAIFuncIEmbed *provider = nullptr;
  ObString access_key;
  ObString model = get_request_model_name();
  int64_t dimension = 0;
  // Bound request size independently of the executor's vector size.
  const int64_t max_batch_size = 32;
  if (!is_dense_embedding_type() || contents.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_NOT_NULL(config)) {
    ObJsonNode *node = config->get_value("dimensions");
    if (OB_NOT_NULL(node)) {
      if (node->json_type() != ObJsonNodeType::J_INT || node->get_int() <= 0) {
        ret = OB_INVALID_ARGUMENT;
      } else {
        dimension = node->get_int();
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObAIFuncUtils::get_embed_provider(*allocator_, endpoint_info_.get_provider(), provider))) {
  } else if (OB_FAIL(endpoint_info_.get_unencrypted_access_key(*allocator_, access_key))) {
  } else if (OB_FAIL(provider->get_header(*allocator_, access_key, headers))) {
  } else {
    for (int64_t start = 0; OB_SUCC(ret) && start < contents.count(); start += max_batch_size) {
      ObArray<ObString> batch;
      ObJsonObject *body = nullptr;
      ObJsonObject *response = nullptr;
      ObIJsonBase *parsed = nullptr;
      ObAIFuncClient client;
      const int64_t end = MIN(start + max_batch_size, contents.count());
      for (int64_t i = start; OB_SUCC(ret) && i < end; ++i) {
        if (contents.at(i).empty()) {
          ret = OB_INVALID_ARGUMENT;
        } else if (OB_FAIL(batch.push_back(contents.at(i)))) {
        }
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(provider->get_body(*allocator_, model, batch, config, body))) {
      } else if (OB_FAIL(client.send_post(*allocator_, endpoint_info_.get_url(), headers, body, response))) {
      } else if (OB_FAIL(provider->parse_output(*allocator_, response, parsed))) {
      } else if (OB_ISNULL(parsed) || parsed->json_type() != ObJsonNodeType::J_ARRAY
                  || parsed->element_count() != batch.count()) {
        ret = OB_INVALID_DATA;
      } else {
        ObJsonArray *array = static_cast<ObJsonArray *>(parsed);
        for (int64_t i = 0; OB_SUCC(ret) && i < batch.count(); ++i) {
          ObJsonNode *embedding = array->get_value(i);
          ObString encoded;
          if (OB_FAIL(validate_embedding(*allocator_, embedding, dimension))) {
          } else if (OB_FAIL(ObAIFuncJsonUtils::print_json_to_str(*allocator_, embedding, encoded))) {
          } else if (OB_FAIL(validated_results.push_back(encoded))) {
          }
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    if (OB_FAIL(results.assign(validated_results))) {
      results.reset();
    }
  }
  return ret;
}

int ObAIFuncUtils::validate_rerank_result(ObIAllocator &allocator, ObJsonArray *response,
                                         int64_t expected_count, int64_t base_index,
                                         ObJsonArray *&result)
{
  int ret = OB_SUCCESS;
  ObJsonArray *validated = nullptr;
  ObArray<int8_t> seen;
  ObArray<ObJsonObject *> ordered;
  result = nullptr;
  if (OB_ISNULL(response) || response->json_type() != ObJsonNodeType::J_ARRAY ||
      expected_count <= 0 || base_index < 0 || base_index > INT64_MAX - expected_count ||
      response->element_count() != expected_count) {
    ret = OB_INVALID_DATA;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, validated))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < expected_count; ++i) {
      if (OB_FAIL(seen.push_back(0))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < expected_count; ++i) {
      ObJsonNode *entry = response->get_value(i);
      ObJsonNode *index_node = nullptr;
      ObJsonNode *score_node = nullptr;
      int64_t index = -1;
      double score = 0;
      if (OB_ISNULL(entry) || entry->json_type() != ObJsonNodeType::J_OBJECT) {
        ret = OB_INVALID_DATA;
      } else {
        ObJsonObject *object = static_cast<ObJsonObject *>(entry);
        index_node = object->get_value("index");
        score_node = object->get_value("relevance_score");
        if (OB_ISNULL(index_node) || OB_ISNULL(score_node)) {
          ret = OB_INVALID_DATA;
        } else if (index_node->json_type() == ObJsonNodeType::J_INT) {
          index = index_node->get_int();
        } else if (index_node->json_type() == ObJsonNodeType::J_UINT &&
                   index_node->get_uint() < static_cast<uint64_t>(expected_count)) {
          index = static_cast<int64_t>(index_node->get_uint());
        } else {
          ret = OB_INVALID_DATA;
        }
        if (OB_SUCC(ret)) {
          const ObJsonNodeType score_type = score_node->json_type();
          if (index < 0 || index >= expected_count || seen.at(index) ||
              (score_type != ObJsonNodeType::J_INT && score_type != ObJsonNodeType::J_UINT &&
               score_type != ObJsonNodeType::J_DOUBLE && score_type != ObJsonNodeType::J_DECIMAL)) {
            ret = OB_INVALID_DATA;
          } else if (OB_FAIL(score_node->to_double(score))) {
            ret = OB_INVALID_DATA;
          } else if (!std::isfinite(score)) {
            ret = OB_INVALID_DATA;
          } else {
            seen.at(index) = 1;
            // Own normalized nodes; do not mutate the provider's response tree.
            ObJsonObject *copy = nullptr;
            ObJsonInt *global_index = nullptr;
            ObJsonDouble *normalized_score = nullptr;
            if (OB_ISNULL(copy = static_cast<ObJsonObject *>(object->clone(&allocator, true)))) {
              ret = OB_ALLOCATE_MEMORY_FAILED;
            } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, base_index + index, global_index))) {
            } else if (OB_ISNULL(normalized_score = OB_NEWx(ObJsonDouble, &allocator, score))) {
              ret = OB_ALLOCATE_MEMORY_FAILED;
            } else if (OB_FAIL(copy->add("index", global_index))) {
            } else if (OB_FAIL(copy->add("relevance_score", normalized_score))) {
            } else if (OB_FAIL(ordered.push_back(copy))) {
            }
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      lib::ob_sort(ordered.begin(), ordered.end(), [](const ObJsonObject *left, const ObJsonObject *right) {
        const double a = left->get_value("relevance_score")->get_double();
        const double b = right->get_value("relevance_score")->get_double();
        return a > b || (a == b && left->get_value("index")->get_int() < right->get_value("index")->get_int());
      });
      for (int64_t i = 0; OB_SUCC(ret) && i < ordered.count(); ++i) {
        if (OB_FAIL(validated->append(ordered.at(i)))) {
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    result = validated;
  }
  return ret;
}

int ObAIFuncModel::call_rerank(ObString &query, ObJsonArray *contents, ObJsonArray *&results)
{
  int ret = OB_SUCCESS;
  ObArray<ObString> headers;
  ObJsonArray *all_results = nullptr;
  ObAIFuncIRerank *provider = nullptr;
  ObString access_key;
  ObString request_model_name = get_request_model_name();
  const int64_t batch_size = 20;
  results = nullptr;
  if (!is_rerank_type() || query.empty() || OB_ISNULL(contents) ||
      contents->element_count() == 0 || !ObAIFuncJsonUtils::ob_is_json_array_all_str(contents)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncUtils::get_rerank_provider(*allocator_, endpoint_info_.get_provider(), provider))) {
  } else if (OB_FAIL(endpoint_info_.get_unencrypted_access_key(*allocator_, access_key))) {
  } else if (OB_FAIL(provider->get_header(*allocator_, access_key, headers))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(*allocator_, all_results))) {
  } else {
    for (int64_t base = 0; OB_SUCC(ret) && base < contents->element_count(); base += batch_size) {
      const int64_t count = std::min(batch_size, static_cast<int64_t>(contents->element_count()) - base);
      ObJsonArray *batch = nullptr;
      ObJsonArray *validated = nullptr;
      ObJsonObject *body = nullptr;
      ObJsonObject *response = nullptr;
      ObJsonObject *config = nullptr;
      ObJsonInt *top_n = nullptr;
      ObIJsonBase *parsed = nullptr;
      ObAIFuncClient client;
      if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(*allocator_, batch))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(*allocator_, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(*allocator_, count, top_n))) {
      } else if (OB_FAIL(config->add("top_n", top_n))) {
      } else {
        for (int64_t j = 0; OB_SUCC(ret) && j < count; ++j) {
          if (OB_FAIL(batch->append(contents->get_value(base + j)))) {
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(provider->get_body(*allocator_, request_model_name, query, batch, config, body))) {
      } else if (OB_FAIL(client.send_post(*allocator_, endpoint_info_.get_url(), headers, body, response))) {
      } else if (OB_FAIL(provider->parse_output(*allocator_, response, parsed))) {
      } else if (OB_ISNULL(parsed) || parsed->json_type() != ObJsonNodeType::J_ARRAY) {
        ret = OB_INVALID_DATA;
      } else if (OB_FAIL(ObAIFuncUtils::validate_rerank_result(*allocator_, static_cast<ObJsonArray *>(parsed),
                                                            count, base, validated))) {
      } else {
        for (int64_t j = 0; OB_SUCC(ret) && j < validated->element_count(); ++j) {
          if (OB_FAIL(all_results->append(validated->get_value(j)))) {
          }
        }
      }
    }
    // Validate the full permutation and order all batches together. Equal scores
    // retain the input order even when a provider returns an unsorted batch.
    if (OB_SUCC(ret)) {
      ret = ObAIFuncUtils::validate_rerank_result(*allocator_, all_results,
                                                contents->element_count(), 0, results);
    }
  }
  if (ret == OB_INVALID_DATA) {
    FORWARD_USER_ERROR(ret, "invalid rerank response: expected one finite score per document");
  }
  return ret;
}

bool ObAIFuncJsonUtils::ob_is_json_array_all_str(ObJsonArray* json_array)
{
  bool is_all_str = true;
  if (OB_ISNULL(json_array)) {
    is_all_str = false;
  }
  for (int64_t i = 0; is_all_str && i < json_array->element_count(); i++) {
    ObJsonNode *node = json_array->get_value(i);
    if (OB_ISNULL(node)) {
      is_all_str = false;
    } else if (node->json_type() != ObJsonNodeType::J_STRING) {
      is_all_str = false;
    }
  }
  return is_all_str;
}

int ObAIFuncPromptObjectUtils::construct_prompt_object(ObIAllocator &allocator, ObString &template_str, ObJsonArray *args_array, ObJsonObject *&prompt_object)
{
  INIT_SUCC(ret);
  ObJsonObject *prompt_obj = NULL;
  ObJsonString *template_json_str = NULL;
  if (template_str.empty() || args_array == NULL) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, template_str, template_json_str))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, prompt_obj))) {
  } else if (OB_FAIL(prompt_obj->add(ObAIFuncPromptObjectUtils::prompt_template_key, template_json_str))) {
  } else if (OB_FAIL(prompt_obj->add(ObAIFuncPromptObjectUtils::prompt_args_key, args_array))) {
  }
  if (OB_SUCC(ret)) {
    prompt_object = prompt_obj;
  }
  return ret;
}

bool ObAIFuncPromptObjectUtils::is_valid_prompt_object(ObJsonObject *prompt_object)
{
  bool is_valid = false;
  if (OB_NOT_NULL(prompt_object) && prompt_object->element_count() == 2) {
    ObJsonNode *template_node = prompt_object->get_value(prompt_template_key);
    ObJsonNode *args_node = prompt_object->get_value(prompt_args_key);
    if (OB_NOT_NULL(template_node) && OB_NOT_NULL(args_node)
        && template_node->json_type() == ObJsonNodeType::J_STRING
        && args_node->json_type() == ObJsonNodeType::J_ARRAY) {
      is_valid = true;
      ObJsonArray *args_array = static_cast<ObJsonArray *>(args_node);
      for (int64_t i = 0; is_valid && i < args_array->element_count(); ++i) {
        ObJsonNode *node = args_array->get_value(i);
        is_valid = OB_NOT_NULL(node)
            && (node->json_type() == ObJsonNodeType::J_STRING
                || node->json_type() == ObJsonNodeType::J_OBJECT
                || node->json_type() == ObJsonNodeType::J_ARRAY);
      }
    }
  }
  return is_valid;
}

namespace
{
int expand_prompt_object(ObIAllocator &allocator, ObJsonObject *prompt_object,
                        ObString &result, const int64_t depth)
{
  int ret = OB_SUCCESS;
  if (depth > JSON_DOCUMENT_MAX_DEPTH) {
    ret = OB_ERR_JSON_OUT_OF_DEPTH;
  } else if (!ObAIFuncPromptObjectUtils::is_valid_prompt_object(prompt_object)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonString *template_node = static_cast<ObJsonString *>(
        prompt_object->get_value(ObAIFuncPromptObjectUtils::prompt_template_key));
    ObJsonArray *args = static_cast<ObJsonArray *>(
        prompt_object->get_value(ObAIFuncPromptObjectUtils::prompt_args_key));
    const ObString text = template_node->get_str();
    ObArray<ObString> rendered_args;
    ObStringBuffer buffer(&allocator);
    if (text.empty()) {
      ret = OB_INVALID_ARGUMENT;
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < args->element_count(); ++i) {
      ObJsonNode *arg = args->get_value(i);
      ObString rendered;
      if (arg->json_type() == ObJsonNodeType::J_STRING) {
        rendered = static_cast<ObJsonString *>(arg)->get_str();
      } else if (arg->json_type() == ObJsonNodeType::J_OBJECT
          && ObAIFuncPromptObjectUtils::is_valid_prompt_object(static_cast<ObJsonObject *>(arg))) {
        ret = expand_prompt_object(allocator, static_cast<ObJsonObject *>(arg), rendered, depth + 1);
      } else {
        ret = ObAIFuncJsonUtils::print_json_to_str(allocator, arg, rendered);
      }
      if (OB_SUCC(ret) && OB_FAIL(rendered_args.push_back(rendered))) {
      }
    }
    int64_t literal_start = 0;
    for (int64_t pos = 0; OB_SUCC(ret) && pos < text.length(); ++pos) {
      if (text.ptr()[pos] != '{') {
        continue;
      }
      int64_t end = pos + 1;
      int64_t index = 0;
      bool overflow = false;
      while (end < text.length() && text.ptr()[end] >= '0' && text.ptr()[end] <= '9') {
        const int digit = text.ptr()[end] - '0';
        if (index > (INT64_MAX - digit) / 10) {
          overflow = true;
        } else if (!overflow) {
          index = index * 10 + digit;
        }
        ++end;
      }
      // Non-numeric braces remain literal, so JSON and other text can be
      // included in templates without escaping every brace.
      if (end > pos + 1 && end < text.length() && text.ptr()[end] == '}') {
        if (overflow || index >= rendered_args.count()) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_prompt: invalid placeholder index");
        } else if (buffer.length() + pos - literal_start + rendered_args[index].length() > INT32_MAX) {
          ret = OB_SIZE_OVERFLOW;
        } else if (OB_FAIL(buffer.append(text.ptr() + literal_start, pos - literal_start))) {
        } else if (OB_FAIL(buffer.append(rendered_args[index]))) {
        } else {
          pos = end;
          literal_start = end + 1;
        }
      }
    }
    if (OB_SUCC(ret)) {
      if (buffer.length() + text.length() - literal_start > INT32_MAX) {
        ret = OB_SIZE_OVERFLOW;
      } else if (OB_FAIL(buffer.append(text.ptr() + literal_start, text.length() - literal_start))) {
      } else {
        ret = buffer.get_result_string(result);
      }
    }
  }
  return ret;
}
} // namespace

int ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(ObIAllocator &allocator,
    ObJsonObject *prompt_object, ObString &replaced_prompt_str)
{
  return expand_prompt_object(allocator, prompt_object, replaced_prompt_str, 0);
}

const ObString ObAIFuncModel::get_request_model_name()
{
  ObString request_model_name = info_.model_;
  if (!endpoint_info_.get_request_model_name().empty()) {
    request_model_name = endpoint_info_.get_request_model_name();
  }
  return request_model_name;
}

} // namespace common

namespace query
{

int ObAIModelResolver::resolve_model_name(
    common::ObIAllocator &allocator,
    const common::ObString &model_id,
    common::ObString &model_name)
{
  int ret = OB_SUCCESS;
  common::ObAIFuncExprInfo *info = nullptr;
  if (OB_FAIL(common::ObAIFuncUtils::get_ai_func_info(
          allocator, model_id, info))) {
  } else if (OB_ISNULL(info)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ob_write_string(allocator, info->model_, model_name))) {
  }
  return ret;
}

} // namespace query
} // namespace oceanbase
