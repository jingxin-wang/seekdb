# AI 函数增强：解题思路与验证

本目录对应「赛题一：AI 函数增强」的六个子题。实现直接位于 seekdb SQL 引擎、Provider 适配层和 Endpoint 管理层；回归测试使用本地 HTTP Mock，不需要真实模型或 API Key。

## 六题的实现要点

| 子题 | 主要修复 | 实现位置（仓库根目录相对路径） |
| --- | --- | --- |
| AI_PROMPT | TEXT/LONGTEXT 用 LOB helper 读取，JSON 数组/对象原样保存在 args；展开时序列化普通 JSON，递归展开嵌套 Prompt；动态缓冲区支持重复占位符并检查下标溢出 | `src/sql/engine/expr/ob_expr_ai/ob_expr_ai_prompt.cpp`、`ob_ai_func_utils.cpp` |
| AI_EMBED 单行 | dimension 必须是正整数；未指定时使用服务默认维度，并检查返回向量维度一致；元素必须是有限数值 | `ob_expr_ai_embed.cpp`、`ob_ai_func_utils.cpp` |
| AI_COMPLETE | 普通字符串、AI_PROMPT、长文本、JSON/文本 config、NULL config；校验响应结构及非空文本 | `ob_expr_ai_complete.cpp`、`ob_ai_func_utils.cpp` |
| Ollama | Provider 白名单和工厂支持大小写兼容；空 Key 合法且数据库 NULL 字段可读取；原生 generate/embed 协议，禁用流式；max_tokens 映射 num_predict | `src/share/ai_service/ob_ai_service_struct.cpp`、`src/observer/ai_service/ob_ai_service_proxy.cpp`、`ob_ai_func_utils.cpp` |
| AI_RERANK | 字符串/对象两条路径统一每批 20 个文档；校验数量、索引和分数，恢复全局 index；按分数降序、原始 index 升序排序；对象保留原字段并加 `_model_score` | `ob_expr_ai_rerank.cpp`、`ob_ai_func_utils.cpp` |
| AI_EMBED 跨行 | 注册 eval_batch_func；跳过过滤行与已求值行，NULL 不发请求；按模型和维度分组，每个 HTTP 请求最多 32 条；按响应 index 还原请求位置，再映射回原始 SQL 行 | `ob_expr_ai_embed.cpp`、`ob_ai_func_utils.cpp` |

`ob_ai_func_client.cpp` 同时修复请求体生命周期、JSON 根类型检查、重试截止时间、错误传播和 Curl 资源释放。Provider 层先完整验证各批结果，再发布结果，避免失败后返回不完整或错位数据。

## 关键语义

- `AI_PROMPT('{0}/{0}', 'abc')` 返回包含 template/args 的 JSON；传给 `AI_COMPLETE` 时展开为 `abc/abc`。
- 普通 JSON 对象/数组作为参数时输出合法 JSON 文本；具有合法 template/args 结构的对象按嵌套 Prompt 展开。
- 占位符沿用从 0 开始的 `{数字}` 形式。越界或整数溢出报错；普通非数字花括号保留为文本，兼容 JSON 字面量。
- `AI_EMBED(model, NULL)` 返回 SQL NULL，且不调用 HTTP 服务。显式 NULL dimension 属于非法维度。
- Embedding 的 JSON 数组必须非空，维度符合显式参数或同次调用默认维度。OpenAI 的 data 数组可以乱序，但 index 必须唯一、完整且在范围内。
- Rerank 不信任服务的排序；即使服务返回乱序、同分顺序不同，也以全局原始 index 保持稳定排序。
- Ollama Endpoint 的 URL 仍由用户指定，例如完整的 `/api/generate` 或 `/api/embed` 地址。

## 构建和单元测试

以下命令在 WSL Ubuntu 中、seekdb 仓库根目录执行：

```bash
python3 build.py build seekdb -j 3
export LD_LIBRARY_PATH="$PWD/deps/3rd/usr/local/oceanbase/deps/devel/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 build.py test //unittest/sql/engine/expr:test_ai_functions -j 3 \
  --test_env=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" --test_output=errors
```

这里的动态库路径用于本地依赖目录中的 libaio。单元测试由以下文件组成：

- `ob_expr_ai_prompt_test.cpp`：7 项，覆盖 JSON、嵌套、Unicode、重复展开、非法下标与非 NUL 结尾字符串。
- `test_ob_openai_utils.cpp`：12 项，覆盖 Provider 请求/响应、乱序索引、非法维度/元素及配置。
- `test_ob_ai_rerank.cpp`：5 项，覆盖 43 个文档的多批稳定排序、异常结果与原始对象映射。

## SQL 端到端测试

在独立本地实例上运行，避免与已有实例共用数据目录。Mock 自动绑定本机空闲端口，脚本会创建唯一命名的数据库、模型和 Endpoint，并在结束时清理它们。

```bash
export LD_LIBRARY_PATH="$PWD/deps/3rd/usr/local/oceanbase/deps/devel/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 tools/deploy/sdb.py start \
  --binary "$PWD/build/bin/src/observer/seekdb" \
  --base-dir "$PWD/build/ai-enhancement-instance" --port 2887 --nodaemon
python3 tools/deploy/sdb.py wait-ready \
  --base-dir "$PWD/build/ai-enhancement-instance" --port 2887 \
  --client "$PWD/deps/3rd/u01/obclient/bin/obclient"
python3 unittest/sql/engine/expr/test_ai_function_enhancement.py \
  --client "$PWD/deps/3rd/u01/obclient/bin/obclient" --port 2887
python3 tools/deploy/sdb.py stop --base-dir "$PWD/build/ai-enhancement-instance"
```

若要重新编译，请先停止实例；构建会替换可执行文件。

2026-10-30 的本地验证结果：seekdb 构建通过；24 个 C++ 单测全部通过；六组 SQL 端到端测试全部通过；原有 `ai_function` mysqltest 套件 6/6 通过，结果见 `build/ai-enhancement-mysqltest-results/seekdb_result.json`。原有 AI_PROMPT 用例中 JSON Object 和 TEXT/BLOB 的“不支持”预期已同步更新。实测 96 行中的 90 条非 NULL 输入形成 `[32, 32, 26]` 三个 HTTP 请求；动态模型/维度、过滤及 LIMIT 场景形成 `[10, 10, 10, 10, 11, 12]` 六个请求。乱序响应、后续批次失败、无部分结果、故障恢复均通过。此结果是本地 Mock 回归结果，不代表比赛官方评分。
