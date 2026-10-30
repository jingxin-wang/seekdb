#!/usr/bin/env python3
"""AI function regression against a running seekdb and a local HTTP fixture.

Usage: python3 test_ai_function_enhancement.py --client /path/to/obclient --port 2881
Only the Python standard library and the SQL client are required. All model and
endpoint names are unique; the created database and AI resources are removed.
"""

import argparse
import json
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def sql_string(value):
    data = str(value).encode("utf-8").hex()
    return "CONVERT(0x%s USING utf8mb4)" % data if data else "''"


def json_sql(value):
    return sql_string(json.dumps(value, ensure_ascii=False))


def vector(text, dimension):
    value = int(text.removeprefix("doc-")) if text.removeprefix("doc-").isdigit() else len(text)
    return [float(value + index / 8) for index in range(dimension)]


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.requests = []
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            return list(self.requests)


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.requests.append({"path": self.path, "body": body,
                                         "authorization": self.headers.get("Authorization")})
        fixture = body.get("model", "")
        status = 200
        response = None
        if fixture == "http-error":
            status, response = 500, {"error": "deliberate fixture failure"}
        elif fixture == "wrong-root":
            response = [1, 2]
        elif self.path in ("/complete", "/api/generate"):
            prompt = body.get("prompt") if self.path == "/api/generate" else body["messages"][-1]["content"]
            value = "" if fixture == "empty-completion" else prompt
            if fixture == "invalid-completion":
                value = {"unexpected": "object"}
            response = ({"response": value, "done": True} if self.path == "/api/generate"
                        else {"choices": [{"message": {"content": value}}]})
        elif self.path in ("/embed", "/api/embed"):
            inputs = body["input"]
            dimension = int(body.get("dimensions", 3))
            data = [{"index": index, "embedding": vector(text, dimension)}
                    for index, text in enumerate(inputs)]
            if fixture == "late-invalid-vector" and "63" in inputs:
                data[inputs.index("63")]["embedding"] = ["invalid"]
            elif fixture == "wrong-dimension":
                data[0]["embedding"].append(99)
            elif fixture == "invalid-vector":
                data[0]["embedding"] = ["not-a-number", {}, None]
            elif fixture == "missing-index":
                del data[0]["index"]
            elif fixture == "duplicate-index" and len(data) > 1:
                data[1]["index"] = data[0]["index"]
            elif fixture == "missing-result":
                data.pop()
            data.reverse()  # OpenAI output indices, not server order, map rows.
            response = ({"embeddings": [vector(text, dimension) for text in inputs]}
                        if self.path == "/api/embed" else {"data": data})
        elif self.path == "/rerank":
            data = [{"index": index, "relevance_score": int(text.removeprefix("doc-")) % 5}
                    for index, text in enumerate(body["documents"])]
            data.sort(key=lambda item: (-item["relevance_score"], -item["index"]))
            if fixture == "bad-rank-index":
                data[0]["index"] = len(data)
            elif fixture == "duplicate-rank-index":
                data[1]["index"] = data[0]["index"]
            elif fixture == "bad-rank-score":
                data[0]["relevance_score"] = "NaN"
            response = {"results": data}
        else:
            status, response = 404, {"error": "unknown route"}
        if fixture == "late-http-error" and "63" in body.get("input", []):
            status, response = 500, {"error": "later batch failure"}
        encoded = b"{bad json" if fixture == "invalid-json" else json.dumps(response).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class Regression:
    def __init__(self, args, server):
        self.args, self.server = args, server
        self.prefix = "ai_reg_" + uuid.uuid4().hex[:10]
        self.database = self.prefix
        self.resources = []
        self.models = {}
        self.passed = 0

    def sql(self, statement, failure=False):
        command = [self.args.client, "--host=" + self.args.host, "--port=" + str(self.args.port),
                   "--user=" + self.args.user, "--default-character-set=utf8mb4",
                   "--batch", "--raw", "--skip-column-names", "--no-auto-rehash"]
        # stdin also exercises LONGTEXT prompts exceeding the OS argument limit.
        query = "SET NAMES utf8mb4;" + statement + ";\n"
        result = subprocess.run(command, input=query, text=True, encoding="utf-8",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if failure:
            assert result.returncode != 0, "expected SQL failure: " + statement[:200]
            assert not result.stdout.strip(), "failed statement returned partial results: " + result.stdout[:200]
            return result.stderr
        if result.returncode != 0:
            raise AssertionError("SQL failed: " + statement[:300] + "\n" + result.stderr)
        return result.stdout.strip().splitlines() if result.stdout.strip() else []

    def check(self, name, callback):
        callback()
        self.passed += 1
        print("PASS " + name, flush=True)

    def create_model(self, key, kind, provider, route):
        name = self.prefix + "_" + key
        endpoint = name + "_ep"
        self.sql("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s,%s)" %
                 (sql_string(name), json_sql({"type": kind, "model_name": "echo"})))
        self.resources.append((name, None))
        params = {"ai_model_name": name, "provider": provider,
                  "url": "http://127.0.0.1:%s%s" % (self.server.server_port, route)}
        if provider.lower() != "ollama":
            params["access_key"] = "local-fixture-only"
        self.sql("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT(%s,%s)" %
                 (sql_string(endpoint), json_sql(params)))
        self.resources[-1] = (name, endpoint)
        self.models[key] = name

    def fixture(self, key, value="echo"):
        self.sql("CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT(%s,%s)" %
                 (sql_string(self.models[key] + "_ep"), json_sql({"request_model_name": value})))

    def complete(self, expression, key="complete", config=None):
        arguments = [sql_string(self.models[key]), expression]
        if config is not None:
            arguments.append(config)
        rows = self.sql("SELECT HEX(AI_COMPLETE(%s))" % ",".join(arguments))
        value = bytes.fromhex(rows[0]).decode("utf-8")
        return value

    def embedding(self, text="7", dimension=3, key="embed"):
        args = [sql_string(self.models[key]), sql_string(text), str(dimension)]
        return json.loads(self.sql("SELECT AI_EMBED(%s)" % ",".join(args))[0])

    def setup(self):
        self.sql("CREATE DATABASE " + self.database)
        for values in [("complete", "completion", "OPENAI", "/complete"),
                       ("ollama", "completion", "oLlAmA", "/api/generate"),
                       ("embed", "dense_embedding", "OPENAI", "/embed"),
                       ("embed2", "dense_embedding", "OPENAI", "/embed"),
                       ("ollama_embed", "dense_embedding", "OLLAMA", "/api/embed"),
                       ("rank", "rerank", "SILICONFLOW", "/rerank")]:
            self.create_model(*values)
        self.sql("CREATE TABLE %s.inputs(id INT PRIMARY KEY, content LONGTEXT, model VARCHAR(128), dim INT)" % self.database)
        values = []
        for index in range(96):
            model = self.models["embed" if index % 2 else "embed2"]
            values.append("(%d,%s,%s,%d)" % (index,
                "NULL" if index % 19 == 0 else sql_string(str(index)), sql_string(model), 2 + index % 3))
        self.sql("INSERT INTO %s.inputs VALUES %s" % (self.database, ",".join(values)))
        self.sql("CREATE TABLE %s.prompts(template LONGTEXT, argument TEXT)" % self.database)
        self.sql("INSERT INTO %s.prompts VALUES (%s,%s)" %
                 (self.database, sql_string("前" * 40000 + "{0}/{0}"), sql_string("文" * 10000)))
        self.sql("CREATE TABLE %s.reverse_prompts(template TEXT, argument LONGTEXT)" % self.database)
        self.sql("INSERT INTO %s.reverse_prompts VALUES ('{0}/{0}',%s)" %
                 (self.database, sql_string("长" * 60000)))

    def prompts(self):
        result = self.complete("AI_PROMPT('list={0}',JSON_ARRAY(1,JSON_OBJECT('x','你好')))")
        assert json.loads(result.removeprefix("list=")) == [1, {"x": "你好"}]
        result = self.complete("AI_PROMPT('{0}',JSON_OBJECT('a',JSON_ARRAY(1,2)))")
        assert json.loads(result) == {"a": [1, 2]}
        nested = "AI_PROMPT('outer {0}', AI_PROMPT('{0}/{0}', %s))" % sql_string('你好\n"\\')
        assert self.complete(nested) == 'outer 你好\n"\\/你好\n"\\'
        rows = self.sql("SELECT HEX(AI_COMPLETE(%s,AI_PROMPT(template,argument))) FROM %s.prompts" %
                        (sql_string(self.models["complete"]), self.database))
        result = bytes.fromhex(rows[0]).decode("utf-8")
        assert result == "前" * 40000 + "文" * 10000 + "/" + "文" * 10000
        rows = self.sql("SELECT HEX(AI_COMPLETE(%s,AI_PROMPT(template,argument))),HEX(AI_COMPLETE(%s,argument)) FROM %s.reverse_prompts" %
                        (sql_string(self.models["complete"]), sql_string(self.models["complete"]), self.database))
        expanded, direct = [bytes.fromhex(value).decode("utf-8") for value in rows[0].split("\t")]
        assert expanded == "长" * 60000 + "/" + "长" * 60000
        assert direct == "长" * 60000
        self.sql("SELECT AI_COMPLETE(%s, AI_PROMPT('{1}','x'))" % sql_string(self.models["complete"]), failure=True)
        self.sql("SELECT AI_COMPLETE(%s, AI_PROMPT('{999999999999999999999999999999}','x'))" %
                 sql_string(self.models["complete"]), failure=True)

    def completions(self):
        assert self.complete(sql_string("plain")) == "plain"
        assert self.complete(sql_string("null config"), config="NULL") == "null config"
        assert self.complete(sql_string("options"), config="JSON_OBJECT('temperature',0.25,'max_tokens',17)") == "options"
        body = self.server.snapshot()[-1]["body"]
        assert body["temperature"] == 0.25 and body["max_tokens"] == 17
        assert self.complete(sql_string("text config"), config=json_sql({"temperature": 0.5})) == "text config"
        self.sql("SELECT AI_COMPLETE(%s,'x',JSON_ARRAY(1))" % sql_string(self.models["complete"]), failure=True)
        for fixture in ("empty-completion", "invalid-completion", "wrong-root", "invalid-json", "http-error"):
            self.fixture("complete", fixture)
            self.sql("SELECT AI_COMPLETE(%s,'x')" % sql_string(self.models["complete"]), failure=True)
        self.fixture("complete")
        assert self.complete(sql_string("after failures")) == "after failures"

    def ollama(self):
        assert self.complete(sql_string("native"), key="ollama", config="JSON_OBJECT('temperature',0.2,'max_tokens',9)") == "native"
        request = self.server.snapshot()[-1]
        assert request["path"] == "/api/generate" and request["authorization"] is None
        assert request["body"]["stream"] is False
        assert request["body"]["options"]["temperature"] == 0.2
        assert request["body"]["options"]["num_predict"] == 9
        self.sql("CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT(%s,%s)" %
                 (sql_string(self.models["ollama"] + "_ep"), json_sql({"provider": "OlLaMa", "access_key": ""})))
        assert self.complete(sql_string("altered"), key="ollama") == "altered"
        assert self.embedding(key="ollama_embed") == vector("7", 3)
        request = self.server.snapshot()[-1]
        assert request["path"] == "/api/embed" and request["body"]["input"] == ["7"]
        self.sql("CALL DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT(%s,%s)" %
                 (sql_string(self.models["ollama"] + "_ep"), json_sql({"provider": "OPENAI"})), failure=True)

    def embeddings(self):
        assert self.embedding() == vector("7", 3)
        assert json.loads(self.sql("SELECT AI_EMBED(%s,'7')" % sql_string(self.models["embed"]))[0]) == vector("7", 3)
        assert self.embedding(dimension=5) == vector("7", 5)
        assert self.sql("SELECT AI_EMBED(%s,NULL)" % sql_string(self.models["embed"])) == ["NULL"]
        for dim in ("0", "-1", "NULL", "1.2"):
            self.sql("SELECT AI_EMBED(%s,'x',%s)" % (sql_string(self.models["embed"]), dim), failure=True)
        for fixture in ("wrong-dimension", "invalid-vector", "missing-index", "missing-result"):
            self.fixture("embed", fixture)
            self.sql("SELECT AI_EMBED(%s,'7',3)" % sql_string(self.models["embed"]), failure=True)
        self.fixture("embed")

    def batches(self):
        before = len(self.server.snapshot())
        rows = self.sql("SELECT id,AI_EMBED(model,content,dim) FROM %s.inputs WHERE MOD(id,7)<>0 ORDER BY id LIMIT 67" % self.database)
        ids = [i for i in range(96) if i % 7 != 0][:67]
        assert len(rows) == len(ids)
        for row, index in zip(rows, ids):
            actual_id, value = row.split("\t")
            assert int(actual_id) == index
            if index % 19 == 0:
                assert value == "NULL"
            else:
                assert json.loads(value) == vector(str(index), 2 + index % 3)
        requests = [item for item in self.server.snapshot()[before:] if item["path"] == "/embed"]
        sizes = [len(item["body"]["input"]) for item in requests]
        assert sizes and max(sizes) > 1, "no real HTTP batch was observed: %r" % sizes
        print("  HTTP embedding batch sizes: " + repr(sizes), flush=True)
        assert all(text is not None for request in requests for text in request["body"]["input"])
        # Same model/dimension spans the HTTP batch limit, while preserving
        # correspondence despite reversed provider response order.
        before = len(self.server.snapshot())
        query = "SELECT id,AI_EMBED(%s,content,3) FROM %s.inputs WHERE content IS NOT NULL ORDER BY id" % (
            sql_string(self.models["embed"]), self.database)
        rows = self.sql(query)
        expected_ids = [i for i in range(96) if i % 19 != 0]
        assert len(rows) == len(expected_ids)
        for row, index in zip(rows, expected_ids):
            actual_id, value = row.split("\t")
            assert int(actual_id) == index and json.loads(value) == vector(str(index), 3)
        requests = [item for item in self.server.snapshot()[before:] if item["path"] == "/embed"]
        sizes = [len(item["body"]["input"]) for item in requests]
        assert len(sizes) >= 3 and max(sizes) <= 32 and sum(sizes) == len(expected_ids), sizes
        print("  HTTP same-model batch sizes: " + repr(sizes), flush=True)
        for fixture in ("late-invalid-vector", "late-http-error"):
            self.fixture("embed", fixture)
            before = len(self.server.snapshot())
            self.sql(query, failure=True)
            requests = [item for item in self.server.snapshot()[before:] if item["path"] == "/embed"]
            assert len(requests) >= 2 and "63" not in requests[0]["body"]["input"]
            assert any("63" in item["body"]["input"] for item in requests)
            self.fixture("embed")
            assert self.embedding() == vector("7", 3)
        # A provider failure in a vectorized query must fail the whole statement.
        self.fixture("embed", "duplicate-index")
        self.sql("SELECT AI_EMBED(%s,content,3) FROM %s.inputs WHERE content IS NOT NULL LIMIT 64" %
                 (sql_string(self.models["embed"]), self.database), failure=True)
        self.fixture("embed")

    def reranking(self):
        documents = ["doc-%d" % i for i in range(47)]
        expected = sorted(range(len(documents)), key=lambda i: (-(i % 5), i))
        expression = "AI_RERANK(%s,'query',%s)" % (sql_string(self.models["rank"]), json_sql(documents))
        result = json.loads(self.sql("SELECT " + expression)[0])
        assert [item["index"] for item in result] == expected
        assert [item["relevance_score"] for item in result] == [i % 5 for i in expected]
        objects = [{"id": 1000 + i, "text": text, "extra": {"original": i}} for i, text in enumerate(documents)]
        expression = "AI_RERANK(%s,'query',%s,'text')" % (sql_string(self.models["rank"]), json_sql(objects))
        result = json.loads(self.sql("SELECT " + expression)[0])
        assert [{key: item[key] for key in ("id", "text", "extra")} for item in result] == [objects[i] for i in expected], "original documents or global ordering changed"
        assert all("_model_score" in item for item in result)
        for fixture in ("bad-rank-index", "duplicate-rank-index", "bad-rank-score"):
            self.fixture("rank", fixture)
            self.sql("SELECT " + expression, failure=True)
        self.fixture("rank")
        self.sql("SELECT AI_RERANK(%s,'query',JSON_ARRAY(JSON_OBJECT('other','x')),'text')" %
                 sql_string(self.models["rank"]), failure=True)

    def cleanup(self):
        errors = []
        for model, endpoint in reversed(self.resources):
            for procedure, name in (("DROP_AI_MODEL_ENDPOINT", endpoint), ("DROP_AI_MODEL", model)):
                if name is not None:
                    try:
                        self.sql("CALL DBMS_AI_SERVICE.%s(%s)" % (procedure, sql_string(name)))
                    except Exception as exc:
                        errors.append(str(exc))
        try:
            self.sql("DROP DATABASE IF EXISTS " + self.database)
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            raise RuntimeError("cleanup failed:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--user", default="root")
    parser.add_argument("--groups", default="prompts,completions,ollama,embeddings,batches,reranking",
                        help="comma-separated independent groups to run")
    args = parser.parse_args()
    server = FixtureServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    regression = Regression(args, server)
    try:
        regression.setup()
        failures = []
        available = {"prompts": ("AI_PROMPT JSON, nested, repeated and LONGTEXT", regression.prompts),
                     "completions": ("AI_COMPLETE config and invalid responses", regression.completions),
                     "ollama": ("Ollama endpoint and native protocols", regression.ollama),
                     "embeddings": ("AI_EMBED scalar and response validation", regression.embeddings),
                     "batches": ("AI_EMBED SQL batches and row mapping", regression.batches),
                     "reranking": ("AI_RERANK batches, ties and original documents", regression.reranking)}
        for group in args.groups.split(","):
            if group not in available:
                raise ValueError("unknown test group: " + group)
            name, callback = available[group]
            try:
                regression.check(name, callback)
            except Exception as exc:
                failures.append((name, exc))
                print("FAIL %s: %s" % (name, exc), flush=True)
            finally:
                # Invalid response fixtures are scoped to one test group.
                for key in regression.models:
                    regression.fixture(key)
        if failures:
            raise AssertionError("%d test groups failed: %s" % (len(failures),
                                 "; ".join("%s: %s" % failure for failure in failures)))
        print("All %d AI enhancement groups passed." % regression.passed, flush=True)
    finally:
        try:
            regression.cleanup()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()
