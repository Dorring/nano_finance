# Phase 0 审核结果：不通过，需整改

独立验证：

- `~/anaconda3/bin/python -m pytest tests/test_phase0.py -q`：26 passed。
- `python -m compileall -q src tests`：通过。
- 但测试中存在与需求相反的断言，且没有覆盖关键攻击输入，因此测试通过不能证明 Phase 0 达标。

## P0：删除操作没有 fail closed

当前 `delete_document_collection(doc_name, user_id=None)` 仍会构造不含租户的过滤器：

- `doc_name` 非空时会删除所有租户的同名文档。
- `doc_name=None` 时会把空过滤器传给 Chroma；测试 fake collection 会清空全库。

当前 `SqliteBM25Retriever.delete_doc(doc_name, user_id=None)` 也显式执行跨租户删除。

更严重的是，`TestFailClosed.test_delete_no_user_deletes_all` 把“无用户时删除全部”作为正确行为，这与任务要求完全相反。

整改要求：

1. 所有 destructive API 在 `user_id is None` 时必须在接触存储前抛出明确异常或返回失败，绝不能删除任何记录。
2. 删除当前用户全部数据必须是一个命名明确、强制接收 `user_id` 的独立函数，不得通过通用删除函数传 `doc_name=None` 隐式实现。
3. 将错误测试改成断言拒绝操作且两个租户数据均保留。

## P0：BM25 使用未转义 LIKE，可扩大查询和删除范围

当前用：

```text
user_{user_id}_{doc_name}::%
```

匹配 `doc_id`。文件名中的 `%` 和 `_` 会被 SQLite 当作通配符。攻击者可构造文件名或 DELETE URL，使查询/删除命中其他文档。

整改要求：

1. BM25 `chunk_store` 增加独立 `doc_name` 列并使用 `c.doc_name = ?` 精确匹配；优先采用此方案。
2. `_init_db` 对已有数据库执行幂等 schema migration，并从 `metadata_json` 回填 `doc_name`；迁移后重建 FTS。
3. 若采用 LIKE 转义，必须显式 `ESCAPE`、正确转义 `%`、`_` 和转义字符，并增加攻击输入测试。但独立列更稳健。
4. 新增文件名包含 `%`、`_`、中文、空格时的搜索和删除隔离测试。

## P0：租户作用域 ID 只在解析器生成，没有在存储边界强制

`add_documents()` 和 `add_chunks()` 接受任意 `metadata.doc_id`。测试通过是因为测试 helper 已经手工生成 `user_*` ID；它没有证明存储函数会阻止未作用域 ID 导致跨租户覆盖。

整改要求：

1. 在 Dense 和 Sparse 的写入边界统一调用同一个纯函数生成/验证 scoped chunk ID。
2. 对同一输入重复调用必须幂等，不能重复加前缀。
3. `user_id=None` 时禁止写入。
4. 测试应向两个存储函数传入相同的未作用域 raw chunk ID，验证最终 ID 不冲突且 Dense/Sparse 一致。
5. 用户 0 也应按合法值处理；判断使用 `is None`，不要依赖 truthiness。

## P1：旧索引兼容只在总结中列为风险，没有实现要求中的处理

现状会让旧 Chroma/BM25 数据静默不可检索。任务要求“安全迁移或明确、可检测的重建策略”，不能只写在总结里。

整改要求：

1. 为索引增加 schema version。
2. 对 BM25 实现幂等迁移。
3. 对无法安全原地迁移的 Chroma 旧索引，在启动/首次访问时明确检测并报出可操作错误，或者提供经过测试的迁移工具。
4. 禁止静默返回空结果伪装成“没有相关文档”。
5. 增加 legacy index 检测/迁移测试。

## P1：清空当前用户接口可能报告虚假成功

`clear_all_documents()` 忽略 `delete_document_collection()` 的布尔返回值；Dense 删除失败后仍删除 BM25 并返回成功，造成双索引不一致。

整改要求：

1. Dense 删除失败时不得返回成功。
2. 明确部分失败行为，至少保留可重试错误并记录哪一侧失败。
3. 添加 Dense 删除异常、BM25 删除异常测试。

## P1：测试质量不足

以下测试只搜索源码字符串，没有执行 endpoint：

- `test_main_query_endpoint_uses_await`
- `test_stream_multi_doc_uses_await`
- 上传和认证相关测试

整改要求：

1. 优先通过 FastAPI endpoint 函数/TestClient 和 mock 验证真实行为。
2. 如果导入依赖过重，至少使用 AST 检查目标调用位于 `Await` 节点，而不是字符串包含。
3. 测试不得在模块导入后永久污染全局 `sys.modules`；使用 fixture/monkeypatch 并恢复。
4. 覆盖异常清理和原始 `HTTPException` 不被统一改写为 500。

## P1：无关文件模式和换行变化

`main.py`、`ingest.py`、`rag_engine.py` 及测试被改成 770/可执行，且修改文件产生整文件 CRLF 差异，`git diff --check` 大量失败。

整改要求：

1. Python 源码和测试恢复普通非可执行模式。
2. 使用 LF，保证 `git diff --check -- finquery_rag` 通过。
3. diff 只保留实际业务修改。
4. 清理 `.cc-work` 不属于 CC 的任务；保留它供后续协调。

## 复验命令

```bash
cd finquery_rag/backend
~/anaconda3/bin/python -m pytest tests/test_phase0.py -v
~/anaconda3/bin/python -m compileall -q src tests
cd ../..
git diff --check -- finquery_rag
git diff --ignore-space-at-eol --stat -- finquery_rag
```

完成后报告：

1. 每个审核项的修改。
2. 新增/修改测试清单。
3. 所有验证命令原始结果。
4. 尚未解决的风险；不得把本轮明确要求再次只列为风险。
