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
#include "ob_expr_ai_rerank.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"
#include "lib/utility/utility.h"
#include "common/json_type/ob_json_common.h"
 
using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{

ObExprAIRerank::ObExprAIRerank(common::ObIAllocator &alloc)
: ObFuncExprOperator(alloc, 
                    T_FUN_SYS_AI_RERANK, 
                    N_AI_RERANK, 
                    MORE_THAN_ZERO,
                    NOT_VALID_FOR_GENERATED_COL, 
                    NOT_ROW_DIMENSION) 
{
}

ObExprAIRerank::~ObExprAIRerank() 
{
}

int ObExprAIRerank::calc_result_typeN(ObExprResType &type,
                                    ObExprResType *types_stack,
                                    int64_t param_num,
                                    common::ObExprTypeCtx &type_ctx) const {
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num < 3 || param_num > 4)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    if (!ob_is_string_type(types_stack[MODEL_IDX].get_type())) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, model key must be string type");
    } else if (!ob_is_string_type(types_stack[QUERY_IDX].get_type())) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, query must be string type");
    } else {
      types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      types_stack[QUERY_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    }
    ObObjType in_type = types_stack[DOCUMENTS_IDX].get_type();
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types_stack, DOCUMENTS_IDX, N_AI_RERANK))) {
    } else if (ob_is_string_type(in_type) && types_stack[DOCUMENTS_IDX].get_collation_type() != CS_TYPE_BINARY) {
      if (types_stack[DOCUMENTS_IDX].get_charset_type() != CHARSET_UTF8MB4) {
        types_stack[DOCUMENTS_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      }
    }
    if (OB_FAIL(ret)) {
    } else if (param_num == 4) {
      ObObjType doc_key_type = types_stack[DOC_KEY_IDX].get_type();
      if (doc_key_type == ObNullType) {
      } else if (ob_is_string_type(doc_key_type)) {
        types_stack[DOC_KEY_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      } else {  
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, doc key must be string type");
      }
    }

    if (OB_SUCC(ret)) {
      type.set_json();
      type.set_length((ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType]).get_length());
    }
  }
  return ret;
}

int ObExprAIRerank::eval_ai_rerank(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res)
{
  int ret = OB_SUCCESS;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  MultimodeAlloctor allocator(tmp_alloc_g.get_allocator());
  lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_RERANK));
  ObString model_id;
  ObString query;
  ObString doc_key;
  bool is_null = false;
  bool has_doc_key = false;
  ObIJsonBase *documents = nullptr;
  ObJsonArray *document_array = nullptr;
  ObJsonArray *text_array = nullptr;
  ObJsonArray *result_array = nullptr;
  ObAIFuncExprInfo *info = nullptr;
  share::ObAiModelEndpointInfo endpoint;
  query::ObIAiEndpointResolver *resolver =
      ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
  res.set_null();
  if (expr.arg_cnt_ != 3 && expr.arg_cnt_ != 4) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObJsonExprHelper::get_json_or_str_data(expr.args_[MODEL_IDX], ctx,
                                                         allocator, model_id, is_null))) {
  } else if (is_null || model_id.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObJsonExprHelper::get_json_or_str_data(expr.args_[QUERY_IDX], ctx,
                                                         allocator, query, is_null))) {
  } else if (is_null || query.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, allocator, DOCUMENTS_IDX,
                                                 documents, is_null))) {
  } else if (is_null || OB_ISNULL(documents) || documents->json_type() != ObJsonNodeType::J_ARRAY) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    document_array = static_cast<ObJsonArray *>(documents);
    if (document_array->element_count() == 0) {
      ret = OB_INVALID_ARGUMENT;
    } else if (expr.arg_cnt_ == 4) {
      if (OB_FAIL(ObJsonExprHelper::get_json_or_str_data(expr.args_[DOC_KEY_IDX], ctx,
                                                       allocator, doc_key, is_null))) {
      } else {
        has_doc_key = !is_null;
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (has_doc_key) {
    ret = get_doc_array_from_documents_array_with_key(allocator, document_array, doc_key, text_array);
  } else if (!ObAIFuncJsonUtils::ob_is_json_array_all_str(document_array)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    text_array = document_array;
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, model_id, info))) {
  } else if (OB_ISNULL(resolver) || OB_ISNULL(info)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(resolver->resolve_by_model_name(model_id, allocator, endpoint))) {
  } else {
    ObAIFuncModel model(allocator, *info, endpoint);
    if (OB_FAIL(model.call_rerank(query, text_array, result_array))) {
    } else if (has_doc_key) {
      ObJsonArray *sorted_documents = nullptr;
      if (OB_FAIL(sort_document_array_by_model_result(allocator, document_array,
                                                     result_array, sorted_documents))) {
      } else {
        result_array = sorted_documents;
      }
    }
    if (OB_SUCC(ret)) {
      ObString raw_str;
      if (OB_ISNULL(result_array)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(result_array->get_raw_binary(raw_str, &allocator))) {
      } else if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, res, raw_str))) {
      }
    }
  }
  return ret;
}

int ObExprAIRerank::get_doc_array_from_documents_array_with_key(ObIAllocator &allocator, ObJsonArray *document_object_array, ObString &doc_key, ObJsonArray *&doc_array)
{
  INIT_SUCC(ret);
  doc_array = nullptr;
  if (OB_ISNULL(document_object_array)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (doc_key.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonArray *doc_array_obj = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, doc_array_obj))) {
    } else {
      int64_t count = document_object_array->element_count();
      ObJsonNode *doc_node = nullptr;
      ObJsonNode *doc_value_node = nullptr;
      ObJsonObject *doc_obj = nullptr;
      for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
        doc_node = document_object_array->get_value(i);
        if (OB_ISNULL(doc_node)) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, document_array element is null");
        } else if (doc_node->json_type() != ObJsonNodeType::J_OBJECT) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, document_array element is not object");
        } else {
          doc_obj = static_cast<ObJsonObject *>(doc_node);
          doc_value_node = doc_obj->get_value(doc_key);
          if (OB_ISNULL(doc_value_node)) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, the value for given doc key is null");
          } else if (doc_value_node->json_type() != ObJsonNodeType::J_STRING) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, the value for given doc key is not string");
          } else if (OB_FAIL(doc_array_obj->append(doc_value_node))) {
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      doc_array = doc_array_obj;
    }
  }
  return ret;
}

int ObExprAIRerank::sort_document_array_by_model_result(ObIAllocator &allocator,
    ObJsonArray *document_array, ObJsonArray *model_result_array, ObJsonArray *&sorted_document_array)
{
  int ret = OB_SUCCESS;
  ObJsonArray *validated = nullptr;
  ObJsonArray *result = nullptr;
  sorted_document_array = nullptr;
  if (OB_ISNULL(document_array)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncUtils::validate_rerank_result(allocator, model_result_array,
                            document_array->element_count(), 0, validated))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, result))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < validated->element_count(); ++i) {
      ObJsonObject *ranking = static_cast<ObJsonObject *>(validated->get_value(i));
      const int64_t index = ranking->get_value(INDEX_KEY)->get_int();
      ObJsonNode *document = document_array->get_value(index);
      ObJsonObject *copy = nullptr;
      if (OB_ISNULL(document) || document->json_type() != ObJsonNodeType::J_OBJECT) {
        ret = OB_INVALID_ARGUMENT;
      } else if (OB_ISNULL(copy = static_cast<ObJsonObject *>(document->clone(&allocator, true)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(copy->add(MODEL_SCORE_KEY, ranking->get_value(SCORE_KEY)))) {
      } else if (OB_FAIL(result->append(copy))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    sorted_document_array = result;
  }
  return ret;
}

int ObExprAIRerank::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                        const ObRawExpr &raw_expr,
                        ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   bool got_data = false;
  //   ObObj const_data;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else {
  //     ObAIFuncExprInfo *info = OB_NEWx(ObAIFuncExprInfo, allocator, *allocator, T_FUN_SYS_AI_RERANK);
  //     if (OB_ISNULL(info)) {
  //       ret = OB_ALLOCATE_MEMORY_FAILED;
  //       LOG_WARN("failed to allocate memory for ai rerank info", K(ret));
  //     } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //       LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //     } else {
  //       rt_expr.extra_info_ = info;
  //     }
  //   }
  // }
  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIRerank::eval_ai_rerank;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
