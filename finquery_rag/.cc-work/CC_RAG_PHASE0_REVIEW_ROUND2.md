# Phase 0 第二轮审核：仍不通过

上一轮整改没有完整落地。不要继续总结；先对照工作树逐项完成。

## 实际复验结果

```text
pytest tests/test_phase0.py -q
1 failed, 25 passed
```

失败测试仍名为 `test_delete_no_user_deletes_all`，说明测试文件没有按审核要求修改。

`git diff --check -- finquery_rag` 仍失败，源码仍为 CRLF/尾随空白，多个 `.py` 文件权限仍是 770。

## 仍未修复的 P0

1. `retrieval.py::search()` 仍使用 `doc_id LIKE user_{id}_{doc_name}::%`，没有使用已经新增的 `doc_name` 列。
2. `retrieval.py::delete_doc()` 仍使用未转义 LIKE。
3. `retrieval.py::delete_doc(user_id=None)` 仍执行跨租户删除；必须在连接数据库前拒绝。
4. `delete_all_for_user(user_id=None)` 没有 fail closed。
5. `_patch_retrieval.py` 遗留在项目根目录，表明补丁脚本没有执行/清理。
6. `chunk_id.is_scoped_chunk_id()` 只检查 `user_`：
   - user 1 提交 `user_2_xxx` 会被当成已作用域 ID。
   - 随后可能覆盖 user 2 的 Chroma/SQLite 主键。
   - 必须验证 ID 属于当前 `user_id`；错误租户前缀应拒绝，不能直接接受。
7. `SCHEMA_VERSION = 2` 只是未使用常量，没有 schema version 表、Chroma 版本检测或 legacy 可操作错误。
8. `clear_all_documents()` 仍忽略 Dense 删除布尔结果，随后删除 BM25 并返回成功。
9. 上传路径仍把业务 `HTTPException(400)` 捕获并改写为 500，没有按要求使用 `finally` 清理并保留原状态码。

## 测试仍未修复

测试文件基本保持旧版本：

- 仍断言无用户删除全部。
- 仍直接永久修改 `sys.modules`。
- 仍用字符串包含检查 `await`、认证和上传。
- 没有 `%`、`_`、中文、空格文件名测试。
- 没有传入未作用域 raw ID 的 Dense/Sparse 一致性测试。
- 没有错误租户 scoped ID 测试。
- 没有 BM25 migration 测试。
- 没有 Dense/BM25 部分失败测试。
- 没有上传 400 状态和异常清理测试。

## 文件卫生仍未修复

必须：

- 删除 `_patch_retrieval.py`。
- 不要新增项目级 `.claude/`；若为本轮临时生成则清理。
- `.py` 文件恢复非可执行模式。
- 将本轮修改文件统一为 LF，`git diff --check -- finquery_rag` 必须零输出且退出码 0。
- 不删除 `.cc-work/`，它是协调目录。

## 必须执行并原样报告

```bash
cd /home/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend
~/anaconda3/bin/python -m pytest tests/test_phase0.py -v
~/anaconda3/bin/python -m compileall -q src tests

cd /home/mxf/projects/Qhhhhhhaaa/nanochat
git diff --check -- finquery_rag
stat -c '%a %n' \
  finquery_rag/backend/src/main.py \
  finquery_rag/backend/src/services/*.py \
  finquery_rag/backend/tests/*.py
```

不得在测试失败、diff check 失败或审核项未落地时报告“Phase 0 完成”。
