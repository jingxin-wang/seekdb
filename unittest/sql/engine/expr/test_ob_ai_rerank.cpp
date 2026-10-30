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
#include <algorithm>
#include <limits>
#include "sql/engine/expr/ob_expr_ai/ob_expr_ai_rerank.h"

namespace oceanbase
{
namespace sql
{
using namespace common;

class TestAIRerank : public ::testing::Test
{
public:
  ObArenaAllocator allocator_;

  ObJsonArray *parse_array(const char *text)
  {
    ObIJsonBase *parsed = nullptr;
    const int ret = ObJsonBaseFactory::get_json_base(&allocator_, ObString(text),
        ObJsonInType::JSON_TREE, ObJsonInType::JSON_TREE, parsed);
    EXPECT_EQ(OB_SUCCESS, ret);
    EXPECT_NE(nullptr, parsed);
    return ret == OB_SUCCESS && parsed->json_type() == ObJsonNodeType::J_ARRAY
        ? static_cast<ObJsonArray *>(parsed) : nullptr;
  }

  int extract(ObJsonArray *documents, const char *key, ObJsonArray *&texts)
  {
    ObString field(key);
    return ObExprAIRerank::get_doc_array_from_documents_array_with_key(
        allocator_, documents, field, texts);
  }

  int map_documents(ObJsonArray *documents, ObJsonArray *ranking, ObJsonArray *&result)
  {
    return ObExprAIRerank::sort_document_array_by_model_result(
        allocator_, documents, ranking, result);
  }

  int64_t index_at(ObJsonArray *ranking, int64_t i)
  {
    return static_cast<ObJsonObject *>(ranking->get_value(i))->get_value("index")->get_int();
  }
};

TEST_F(TestAIRerank, unsorted_response_and_equal_scores_follow_input_order)
{
  ObJsonArray *response = parse_array(R"([{"index":3,"relevance_score":0.2},
      {"index":2,"relevance_score":1},{"index":0,"relevance_score":1.0},
      {"index":1,"relevance_score":0.5}])");
  ObJsonArray *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_rerank_result(allocator_, response, 4, 20, result));
  ASSERT_EQ(4, result->element_count());
  EXPECT_EQ(20, index_at(result, 0));
  EXPECT_EQ(22, index_at(result, 1));
  EXPECT_EQ(21, index_at(result, 2));
  EXPECT_EQ(23, index_at(result, 3));
  // Validating a response must not rewrite local indices in the input tree.
  EXPECT_EQ(3, index_at(response, 0));
}

TEST_F(TestAIRerank, rejects_missing_duplicate_out_of_range_and_malformed_values)
{
  const char *invalid[] = {
      R"([])",
      R"([{"index":0,"relevance_score":0.1}])",
      R"([{"index":0,"relevance_score":0.1},{"index":0,"relevance_score":0.2}])",
      R"([{"index":-1,"relevance_score":0.1},{"index":1,"relevance_score":0.2}])",
      R"([{"index":2,"relevance_score":0.1},{"index":1,"relevance_score":0.2}])",
      R"([{"index":0.0,"relevance_score":0.1},{"index":1,"relevance_score":0.2}])",
      R"([{"index":"0","relevance_score":0.1},{"index":1,"relevance_score":0.2}])",
      R"([{"index":0,"relevance_score":"0.1"},{"index":1,"relevance_score":0.2}])",
      R"([{"index":0,"relevance_score":true},{"index":1,"relevance_score":0.2}])",
      R"([{"index":0},{"index":1,"relevance_score":0.2}])",
      R"([null,{"index":1,"relevance_score":0.2}])",
      R"([[],{"index":1,"relevance_score":0.2}])"};
  for (const char *text : invalid) {
    ObJsonArray *result = nullptr;
    EXPECT_EQ(OB_INVALID_DATA, ObAIFuncUtils::validate_rerank_result(
        allocator_, parse_array(text), 2, 0, result)) << text;
    EXPECT_EQ(nullptr, result);
  }
}

TEST_F(TestAIRerank, rejects_nonfinite_scores_and_overflowing_offsets)
{
  ObJsonArray *response = parse_array(R"([{"index":0,"relevance_score":1}])");
  ObJsonObject *entry = static_cast<ObJsonObject *>(response->get_value(0));
  for (double value : {std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    ObJsonDouble score(value);
    ASSERT_EQ(OB_SUCCESS, entry->add("relevance_score", &score));
    ObJsonArray *result = nullptr;
    EXPECT_EQ(OB_INVALID_DATA, ObAIFuncUtils::validate_rerank_result(allocator_, response, 1, 0, result));
    EXPECT_EQ(nullptr, result);
  }
  ObJsonArray *result = nullptr;
  EXPECT_EQ(OB_INVALID_DATA, ObAIFuncUtils::validate_rerank_result(allocator_, response, 1, INT64_MAX, result));
}

TEST_F(TestAIRerank, more_than_two_batches_are_sorted_globally)
{
  ObJsonArray *all = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, all));
  for (int64_t base = 0; base < 43; base += 20) {
    const int64_t count = std::min<int64_t>(20, 43 - base);
    ObJsonArray *batch = nullptr;
    ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_array(allocator_, batch));
    for (int64_t i = count - 1; i >= 0; --i) {
      ObJsonObject *entry = nullptr;
      ObJsonInt *index = nullptr;
      ObJsonInt *score = nullptr;
      ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_object(allocator_, entry));
      ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_int(allocator_, i, index));
      ASSERT_EQ(OB_SUCCESS, ObAIFuncJsonUtils::get_json_int(allocator_, (base + i) % 3, score));
      ASSERT_EQ(OB_SUCCESS, entry->add("index", index));
      ASSERT_EQ(OB_SUCCESS, entry->add("relevance_score", score));
      ASSERT_EQ(OB_SUCCESS, batch->append(entry));
    }
    ObJsonArray *validated = nullptr;
    ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_rerank_result(allocator_, batch, count, base, validated));
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_EQ(OB_SUCCESS, all->append(validated->get_value(i)));
    }
  }
  ObJsonArray *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncUtils::validate_rerank_result(allocator_, all, 43, 0, result));
  ASSERT_EQ(43, result->element_count());
  int64_t position = 0;
  for (int64_t score = 2; score >= 0; --score) {
    for (int64_t index = score; index < 43; index += 3) {
      EXPECT_EQ(index, index_at(result, position++));
    }
  }
}

TEST_F(TestAIRerank, object_text_selection_and_original_ids_are_preserved)
{
  ObJsonArray *documents = parse_array(R"([{"id":"A","text":"first","extra":17},
      {"id":"B","text":"second"},{"id":"C","text":"third"}])");
  ObJsonArray *texts = nullptr;
  ASSERT_EQ(OB_SUCCESS, extract(documents, "text", texts));
  ASSERT_EQ(3, texts->element_count());
  EXPECT_EQ(ObString("first"), ObString(texts->get_value(0)->get_data_length(), texts->get_value(0)->get_data()));
  ObJsonArray *ranking = parse_array(R"([{"index":2,"relevance_score":0.5},
      {"index":0,"relevance_score":0.5},{"index":1,"relevance_score":0.9}])");
  ObJsonArray *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, map_documents(documents, ranking, result));
  ASSERT_EQ(3, result->element_count());
  const char *ids[] = {"B", "A", "C"};
  for (int64_t i = 0; i < 3; ++i) {
    ObJsonObject *entry = static_cast<ObJsonObject *>(result->get_value(i));
    ObJsonNode *id = entry->get_value("id");
    EXPECT_EQ(ObString(ids[i]), ObString(id->get_data_length(), id->get_data()));
    EXPECT_NE(nullptr, entry->get_value("_model_score"));
  }
  EXPECT_EQ(17, static_cast<ObJsonObject *>(result->get_value(1))->get_value("extra")->get_int());
  EXPECT_EQ(nullptr, static_cast<ObJsonObject *>(documents->get_value(0))->get_value("_model_score"));
  EXPECT_EQ(OB_INVALID_ARGUMENT, extract(documents, "missing", texts));
  EXPECT_EQ(OB_INVALID_ARGUMENT, extract(documents, "", texts));
  EXPECT_EQ(OB_INVALID_ARGUMENT, extract(parse_array(R"([{"text":4}])"), "text", texts));
}

} // namespace sql
} // namespace oceanbase
