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
#include <string>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

namespace oceanbase
{
namespace common
{
namespace
{
class PromptTest : public ::testing::Test
{
protected:
  ObJsonObject *parse_prompt(const char *json)
  {
    ObString text = ObString::make_string(json);
    ObIJsonBase *value = nullptr;
    EXPECT_EQ(OB_SUCCESS, ObJsonBaseFactory::get_json_base(&allocator_, text,
        ObJsonInType::JSON_TREE, ObJsonInType::JSON_TREE, value));
    if (value == nullptr || value->json_type() != ObJsonNodeType::J_OBJECT) {
      ADD_FAILURE() << "expected a JSON object";
      return nullptr;
    }
    return static_cast<ObJsonObject *>(value);
  }

  int expand(const char *json, ObString &result)
  {
    return ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
        allocator_, parse_prompt(json), result);
  }

  std::string string(const ObString &value)
  {
    return value.empty() ? std::string() : std::string(value.ptr(), value.length());
  }

  ObArenaAllocator allocator_;
};

TEST_F(PromptTest, repeats_large_arguments_without_truncating)
{
  const std::string long_text(128 * 1024, 'x');
  ObString arg(long_text.length(), long_text.data());
  ObString text = ObString::make_string("{0}/{0}/{0}");
  ObJsonArray *args = nullptr;
  ObJsonString *value = nullptr;
  ObJsonObject *prompt = nullptr;
  ObString result;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, args));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_string(allocator_, arg, value));
  ASSERT_EQ(OB_SUCCESS, args->append(value));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::construct_prompt_object(
      allocator_, text, args, prompt));
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt, result));
  EXPECT_EQ(long_text + "/" + long_text + "/" + long_text, string(result));
}

TEST_F(PromptTest, nested_prompts_and_unicode_escapes)
{
  ObString result;
  ASSERT_EQ(OB_SUCCESS, expand(
      R"({"template":"outer {0}","args":[{"template":"{0}/{0}","args":["\u4f60\u597d\n\"\\"]}]})",
      result));
  EXPECT_EQ("outer 你好\n\"\\/你好\n\"\\", string(result));
}

TEST_F(PromptTest, json_arrays_and_objects_are_serialized_as_json)
{
  const char *inputs[] = {
    R"({"template":"{0}","args":[[1,{"text":"\u4f60\u597d"},null]]})",
    R"({"template":"{0}","args":[{"text":"quote \" and slash \\"}]})",
  };
  for (const char *input : inputs) {
    ObString result;
    ASSERT_EQ(OB_SUCCESS, expand(input, result));
    ObIJsonBase *rendered = nullptr;
    ASSERT_EQ(OB_SUCCESS, ObJsonBaseFactory::get_json_base(&allocator_, result,
        ObJsonInType::JSON_TREE, ObJsonInType::JSON_TREE, rendered));
    ASSERT_NE(nullptr, rendered);
    ObJsonObject *prompt = parse_prompt(input);
    ObJsonArray *args = static_cast<ObJsonArray *>(prompt->get_value("args"));
    ObString expected;
    ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::print_json_to_str(allocator_, args->get_value(0), expected));
    EXPECT_EQ(string(expected), string(result));
  }
}

TEST_F(PromptTest, invalid_indices_do_not_overflow_or_access_out_of_bounds)
{
  ObString result;
  EXPECT_EQ(OB_INVALID_ARGUMENT, expand(R"({"template":"{1}","args":["x"]})", result));
  EXPECT_EQ(OB_INVALID_ARGUMENT, expand(R"({"template":"{9223372036854775808}","args":["x"]})", result));
  EXPECT_EQ(OB_INVALID_ARGUMENT, expand(R"({"template":"{9999999999999999999999999999999999999}","args":["x"]})", result));
  EXPECT_EQ(OB_INVALID_ARGUMENT, expand(R"({"template":"{0}","args":[]})", result));
}

TEST_F(PromptTest, literal_braces_and_empty_arguments)
{
  ObString result;
  ASSERT_EQ(OB_SUCCESS, expand(R"({"template":"{} {name} {0} {0} {","args":[""]})", result));
  EXPECT_EQ("{} {name}   {", string(result));
}

TEST_F(PromptTest, rejects_invalid_prompt_shapes)
{
  const char *invalid[] = {
    R"({"template":"x","args":{}})",
    R"({"template":[],"args":[]})",
    R"({"template":"x","args":[3]})",
    R"({"template":"x","args":[null]})",
    R"({"template":"x","args":[],"extra":1})",
  };
  ObString result;
  for (const char *input : invalid) {
    EXPECT_FALSE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(parse_prompt(input)));
    EXPECT_EQ(OB_INVALID_ARGUMENT, expand(input, result));
  }
  EXPECT_EQ(OB_INVALID_ARGUMENT, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, nullptr, result));
}

TEST_F(PromptTest, meta_prompt_search_respects_string_length)
{
  char source[] = {'a', '\0', '{', '0', '}', 'z', '{', '0', '}'};
  char key_bytes[] = {'{', '0', '}', '!'};
  ObString text(6, source);
  ObString key(3, key_bytes);
  ObString replacement = ObString::make_string("ok");
  ObString result;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptUtils::replace_meta_prompt(
      allocator_, text, key, replacement, result));
  EXPECT_EQ(std::string("a\0okz", 5), string(result));
  ObString absent(2, source);
  EXPECT_EQ(OB_ERR_UNEXPECTED, ObAIFuncPromptUtils::replace_meta_prompt(
      allocator_, absent, key, replacement, result));
}
} // namespace
} // namespace common
} // namespace oceanbase
