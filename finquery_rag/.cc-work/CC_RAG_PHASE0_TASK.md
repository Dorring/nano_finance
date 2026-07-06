# CC 执行任务：FinQuery RAG Phase 0

请直接实现，不要只输出方案。

仓库：`Y:\nanochat`

开始前完整阅读：

- `Y:\nanochat\.claude\AGENTS.md`
- `Y:\nanochat\.claude\RAG_PRODUCTION_OPTIMIZATION_PLAN.md`

## 工作范围

仅允许修改 `finquery_rag/`。当前工作树包含大量用户未提交修改：

- 不得修改其他目录。
- 不得 reset、checkout、clean、stash 或覆盖已有改动。
- 不得创建 commit。
- 不得进行无关重构或大规模格式化。

## 必须完成

1. 修复 `query_collection` 调用参数顺序。调用时优先使用关键字参数，防止再次发生顺序错误。
2. 修复 `/query` 未 await `RAGEngine.query()`。
3. 修复流式多文档路径未 await `retrieve_multiple_documents()`。
4. 解决跨租户同名文件覆盖：
   - Chroma ID 和 SQLite 主键必须包含租户作用域。
   - Dense 与 Sparse 必须使用一致的最终 chunk ID。
   - 不得只依赖 metadata 中的 `user_id`。
5. 所有检索、列表、统计和删除入口必须 fail closed：
   - 缺失 `user_id` 时不得查询或删除全库。
   - BM25 `delete_doc` 必须同时限制用户和文档。
6. 将未认证的全局 `DELETE /documents` 改为仅清理当前登录用户的数据；Dense 与 Sparse 均须清理。
7. 上传临时文件不得直接拼接用户文件名，避免路径穿越；确保成功和异常路径都清理临时文件。
8. 增加 CPU-safe、无网络单元测试，至少覆盖：
   - 向量查询参数和过滤条件正确。
   - `/query` 正确 await。
   - 流式多文档检索正确 await。
   - 两个用户上传同名文件不会覆盖。
   - 一个用户删除同名文件不会影响另一个用户。
   - 缺失用户上下文时查询和删除被拒绝。
   - 当前用户清空文档不会影响其他用户。

## 设计要求

- 尽量保持公开 API 响应兼容。
- 对现有 SQLite 表结构变更要考虑已有本地数据库；提供安全迁移或明确、可检测的重建策略。
- 避免在 FastAPI async handler 中执行不必要的阻塞操作；本阶段可保持现有总体结构，不做大重构。
- 不吞掉关键异常；客户端错误和服务端错误应区分。
- 测试不得加载真实 sentence-transformer 模型、调用外部 LLM 或要求 GPU。

## 验证

在 `finquery_rag/backend` 下执行可用的测试命令。若依赖或环境阻止测试，准确记录命令和错误，不要伪造成功。

完成后输出：

1. 修改文件列表。
2. 每项问题的修复方式。
3. 执行的测试命令及结果。
4. 仍存在的风险或未完成项。
