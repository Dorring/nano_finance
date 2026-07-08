#!/usr/bin/env python3
import argparse
import json


def convert_line(obj):
    # If already a list of messages, return as-is
    if isinstance(obj, list):
        return obj
    # If dict with keys 'user' and 'assistant', wrap into expected list
    if isinstance(obj, dict):
        user = obj.get('user') or obj.get('prompt') or obj.get('instruction')
        assistant = obj.get('assistant') or obj.get('answer') or obj.get('response')
        if user is None or assistant is None:
            raise ValueError(f"Unrecognized dict format: missing user/assistant keys: {list(obj.keys())}")
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    raise ValueError(f"Unsupported JSON object type: {type(obj)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', required=True)
    args = p.parse_args()

    total = 0
    with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            new = convert_line(obj)
            fout.write(json.dumps(new, ensure_ascii=False) + '\n')
            total += 1
    print(f"Converted {total} lines from {args.input} -> {args.output}")


if __name__ == '__main__':
    main()
