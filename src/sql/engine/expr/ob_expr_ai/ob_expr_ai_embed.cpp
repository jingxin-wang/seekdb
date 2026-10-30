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
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_type(
        ob_is_text_tc(types_stack[CONTENT_IDX].get_type()) ? ObLongTextType : ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null() || arg_content->is_null()) {
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id = arg_model_id->get_string();
    ObString content;
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_, temp_allocator,
          *arg_content, expr.args_[CONTENT_IDX]->datum_meta_,
          expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), content))) {
    } else if (model_id.empty() || content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    }
    int64_t dim = 0;
    ObJsonInt *dim_json = nullptr;
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_NOT_NULL(arg_dim)) {
      dim = arg_dim->is_null() ? 0 : arg_dim->get_int();
      if (dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        res.set_null();
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
      } else if (OB_FAIL(config->add("dimensions", dim_json))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                      const ObBitVector &skip, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  MultimodeAlloctor allocator(tmp_alloc_g.get_allocator());
  ObEvalCtx::BatchInfoScopeGuard batch_guard(ctx);
  batch_guard.set_batch_size(batch_size);
  ObBitVector &evaluated = expr.get_evaluated_flags(ctx);
  ObDatum *datums = expr.locate_batch_datums(ctx);
  ObArray<int64_t> rows;
  ObArray<int64_t> dimensions;
  ObArray<ObString> models;
  ObArray<ObString> contents;
  ObArray<ObString> outputs;
  // A NULL content never reaches the provider. Preserve the original row numbers
  // separately from request positions, including skipped and already evaluated rows.
  for (int64_t arg = 0; OB_SUCC(ret) && arg < expr.arg_cnt_; ++arg) {
    if (OB_FAIL(expr.args_[arg]->eval_batch(ctx, skip, batch_size))) {
    }
  }
  for (int64_t row = 0; OB_SUCC(ret) && row < batch_size; ++row) {
    if (skip.at(row) || evaluated.at(row)) {
      continue;
    }
    batch_guard.set_batch_idx(row);
    ObDatum *model = expr.args_[MODEL_IDX]->locate_expr_datumvector(ctx).at(row);
    ObDatum *content = expr.args_[CONTENT_IDX]->locate_expr_datumvector(ctx).at(row);
    ObDatum *dim = expr.arg_cnt_ == 3
        ? expr.args_[DIM_IDX]->locate_expr_datumvector(ctx).at(row) : nullptr;
    ObString input;
    int64_t dimension = dim == nullptr ? 0 : (dim->is_null() ? 0 : dim->get_int());
    if (model->is_null() || content->is_null()) {
      datums[row].set_null();
    } else if (dim != nullptr && dimension <= 0) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
    } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_, allocator,
                   *content, expr.args_[CONTENT_IDX]->datum_meta_,
                   expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), input))) {
    } else if (model->get_string().empty() || input.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
    } else if (OB_FAIL(rows.push_back(row))) {
    } else if (OB_FAIL(dimensions.push_back(dimension))) {
    } else if (OB_FAIL(models.push_back(model->get_string()))) {
    } else if (OB_FAIL(contents.push_back(input))) {
    } else if (OB_FAIL(outputs.push_back(ObString()))) {
    }
  }
  // Model and dimension can themselves be columns. Only batch compatible rows.
  for (int64_t first = 0; OB_SUCC(ret) && first < rows.count(); ++first) {
    if (!outputs.at(first).empty()) {
      continue;
    }
    ObArray<ObString> inputs;
    ObArray<int64_t> positions;
    ObArray<ObString> results;
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo endpoint;
    ObJsonObject *config = nullptr;
    query::ObIAiEndpointResolver *resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    for (int64_t i = first; OB_SUCC(ret) && i < rows.count(); ++i) {
      if (outputs.at(i).empty() && models.at(i) == models.at(first)
          && dimensions.at(i) == dimensions.at(first)) {
        if (OB_FAIL(inputs.push_back(contents.at(i)))) {
        } else if (OB_FAIL(positions.push_back(i))) {
        }
      }
    }
    if (OB_SUCC(ret) && dimensions.at(first) > 0) {
      ObJsonInt *dimension = nullptr;
      if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, dimensions.at(first), dimension))) {
      } else if (OB_FAIL(config->add("dimensions", dimension))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, models.at(first), info))) {
    } else if (OB_FAIL(resolver->resolve_by_model_name(models.at(first), allocator, endpoint))) {
    } else {
      ObAIFuncModel model(allocator, *info, endpoint);
      if (OB_FAIL(model.call_dense_embedding_vector_v2(inputs, config, results))) {
      } else if (results.count() != positions.count()) {
        ret = OB_INVALID_DATA;
      } else {
        for (int64_t i = 0; i < results.count(); ++i) {
          outputs.at(positions.at(i)) = results.at(i);
        }
      }
    }
  }
  // Publish only after every request succeeds; no partial batch is marked valid.
  for (int64_t i = 0; OB_SUCC(ret) && i < rows.count(); ++i) {
    batch_guard.set_batch_idx(rows.at(i));
    if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, datums[rows.at(i)], outputs.at(i)))) {
    }
  }
  if (OB_SUCC(ret)) {
    for (int64_t row = 0; row < batch_size; ++row) {
      if (!skip.at(row)) {
        evaluated.set(row);
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx,
                         const ObRawExpr &raw_expr,
                         ObExpr &rt_expr) const
{
  UNUSED(expr_cg_ctx);
  UNUSED(raw_expr);
  rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
  rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  return OB_SUCCESS;
}

} // namespace sql
} // namespace oceanbase
