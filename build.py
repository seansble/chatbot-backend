# backend/build.py
#!/usr/bin/env python
"""
Knowledge base 빌드 스크립트
Usage: python build.py [--rebuild] [--source data/documents]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict

# RAG 모듈 임포트
sys.path.append(str(Path(__file__).parent))
from rag.tokenizer import KiwiTokenizer


class KnowledgeBuilder:
    def __init__(self, source_dir: str = "data/documents"):
        self.source_dir = Path(source_dir)
        self.output_path = Path("data/knowledge_kiwi.json")
        self.tokenizer = KiwiTokenizer()

    def load_documents(self) -> List[Dict]:
        """documents 폴더에서 모든 문서 로드"""
        documents = []

        # .txt, .json, .md 파일 모두 로드
        for file_path in self.source_dir.glob("**/*"):
            if file_path.suffix in [".txt", ".json", ".md"]:
                print(f"📄 Loading: {file_path.name}")

                if file_path.suffix == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            documents.extend(data)
                        else:
                            documents.append(data)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append(
                            {
                                "text": content,
                                "source": file_path.name,
                                "metadata": {"file_type": file_path.suffix},
                            }
                        )

        return documents

    def build_knowledge(self, documents: List[Dict]) -> List[Dict]:
        """문서를 토큰화하여 knowledge base 생성"""
        knowledge = []

        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"🔄 Processing... {i}/{len(documents)}")

            # 텍스트 추출
            text = doc.get("text", "")
            if not text:
                continue

            # Kiwi로 토큰화
            tokens = self.tokenizer.tokenize(text)

            # Knowledge 항목 생성
            item = {
                "id": doc.get("id", f"doc_{i}"),
                "text": text,
                "tokens": tokens,
                "source": doc.get("source", "unknown"),
                "metadata": doc.get("metadata", {}),
            }

            knowledge.append(item)

            # 첫 3개 샘플 출력
            if i < 3:
                print(f"  Sample {i+1}: {tokens[:10]}...")

        return knowledge

    def save_knowledge(self, knowledge: List[Dict]):
        """knowledge를 JSON 파일로 저장"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved to: {self.output_path}")
        print(f"✅ Total {len(knowledge)} documents processed")

    def run(self, rebuild: bool = False):
        """빌드 실행"""
        print("🚀 Knowledge Base Builder")
        print("=" * 50)

        # 기존 파일 체크
        if self.output_path.exists() and not rebuild:
            print(f"⚠️  {self.output_path} already exists!")
            print("Use --rebuild to overwrite")
            return

        # 문서 로드
        print(f"📂 Loading documents from: {self.source_dir}")
        documents = self.load_documents()

        if not documents:
            print("❌ No documents found!")
            return

        print(f"📊 Found {len(documents)} documents")

        # Knowledge 빌드
        print("\n🔧 Building knowledge base with Kiwi tokenizer...")
        knowledge = self.build_knowledge(documents)

        # 저장
        self.save_knowledge(knowledge)

        print("\n✨ Build complete!")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge base")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild even if output exists"
    )
    parser.add_argument(
        "--source", default="data/documents", help="Source directory for documents"
    )

    args = parser.parse_args()

    builder = KnowledgeBuilder(source_dir=args.source)
    builder.run(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
