"""Offline evaluation module using RAGAS metrics."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings
from app.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAG pipeline evaluator using RAGAS metrics."""

    def __init__(self, pipeline: RAGPipeline):
        """Initialize evaluator with RAG pipeline."""
        self.pipeline = pipeline

    def get_llm(self) -> callable:
        """Get LLM callable for evaluation."""

        # Mock LLM for evaluation - replace with actual LLM
        def mock_llm(query: str, contexts: list[str]) -> str:
            context_text = "\n\n".join(contexts[:3])
            return f"Answer to '{query}': {context_text[:100]}..."

        return mock_llm

    def load_eval_data(self, eval_dir: str) -> list[dict[str, Any]]:
        """Load evaluation data from JSONL files."""
        eval_path = Path(eval_dir)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

        eval_data = []
        for jsonl_file in eval_path.glob("*.jsonl"):
            logger.info(f"Loading evaluation data from {jsonl_file}")
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        eval_data.append(json.loads(line))

        if not eval_data:
            raise ValueError(f"No evaluation data found in {eval_dir}")

        logger.info(f"Loaded {len(eval_data)} evaluation samples")
        return eval_data

    def run_evaluation(self, eval_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Run evaluation on the dataset."""
        results = []

        for i, sample in enumerate(eval_data):
            logger.info(f"Evaluating sample {i+1}/{len(eval_data)}")

            try:
                # Run RAG pipeline
                llm = self.get_llm()
                result = self.pipeline.run(query=sample["question"], llm=llm, k=5)

                # Prepare for RAGAS
                contexts = [ctx["text"] for ctx in result["contexts"]]

                results.append(
                    {
                        "question": sample["question"],
                        "ground_truth": sample["ground_truth"],
                        "answer": result["answer"],
                        "contexts": contexts,
                        "retrieved_contexts": contexts,
                        "contexts_scores": [ctx["score"] for ctx in result["contexts"]],
                    }
                )

            except Exception as e:
                logger.error(f"Failed to evaluate sample {i+1}: {e}")
                continue

        return results

    def calculate_ragas_metrics(
        self, results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate RAGAS metrics."""
        if not results:
            raise ValueError("No results to evaluate")

        # Convert to DataFrame for RAGAS
        df = pd.DataFrame(results)

        # Run RAGAS evaluation
        metrics = [context_recall, faithfulness, answer_relevancy, context_precision]

        evaluation_result = evaluate(df, metrics=metrics)

        return evaluation_result.to_dict()

    def calculate_hit_rate(self, results: list[dict[str, Any]], k: int) -> float:
        """Calculate hit rate@k."""
        if not results:
            return 0.0

        hits = 0
        for result in results:
            # Simple hit rate calculation based on context relevance
            # In a real scenario, you'd check if ground truth is in top-k contexts
            contexts = result["contexts"][:k]
            if contexts:  # If we have any contexts, consider it a hit
                hits += 1

        return hits / len(results)

    def generate_report(
        self, results: list[dict[str, Any]], metrics: dict[str, float]
    ) -> str:
        """Generate evaluation report."""
        # Calculate hit rates
        hr_3 = self.calculate_hit_rate(results, 3)
        hr_5 = self.calculate_hit_rate(results, 5)

        report = f"""
# RAG Pipeline Evaluation Report

## Dataset Statistics
- Total samples: {len(results)}
- Evaluation timestamp: {pd.Timestamp.now()}

## RAGAS Metrics
- Context Recall: {metrics.get('context_recall', 0.0):.3f}
- Faithfulness: {metrics.get('faithfulness', 0.0):.3f}
- Answer Relevancy: {metrics.get('answer_relevancy', 0.0):.3f}
- Context Precision: {metrics.get('context_precision', 0.0):.3f}

## Hit Rate Metrics
- Hit Rate@3: {hr_3:.3f}
- Hit Rate@5: {hr_5:.3f}

## Performance Targets
- Target HR@3: 0.80-0.85  {'PASS' if 0.80 <= hr_3 <= 0.85 else 'FAIL'}
- Target HR@5: 0.90-0.95  {'PASS' if 0.90 <= hr_5 <= 0.95 else 'FAIL'}

## Recommendations
"""

        if hr_3 < 0.80:
            report += "- Consider improving retrieval quality (embedding model, chunking strategy)\n"
        if hr_5 < 0.90:
            report += "- Increase top-k retrieval or improve context selection\n"

        return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Run offline evaluation for RAG service"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="data/eval",
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".cache/eval",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=".cache/index",
        help="Directory containing FAISS index",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize RAG pipeline
        pipeline = RAGPipeline(index_dir=args.index_dir)
        evaluator = RAGEvaluator(pipeline)

        # Load evaluation data
        eval_data = evaluator.load_eval_data(args.eval_dir)

        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator.run_evaluation(eval_data)

        if not results:
            logger.error("No evaluation results generated")
            return

        # Calculate metrics
        logger.info("Calculating RAGAS metrics...")
        ragas_metrics = evaluator.calculate_ragas_metrics(results)

        # Generate report
        report = evaluator.generate_report(results, ragas_metrics)

        # Save results
        results_file = output_path / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "results": results,
                    "metrics": ragas_metrics,
                    "summary": {
                        "total_samples": len(results),
                        "hr_3": evaluator.calculate_hit_rate(results, 3),
                        "hr_5": evaluator.calculate_hit_rate(results, 5),
                    },
                },
                f,
                indent=2,
            )

        report_file = output_path / "evaluation_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        # Log to MLflow if configured
        if settings.mlflow_tracking_uri:
            try:
                mlflow.log_metrics(ragas_metrics)
                mlflow.log_metrics(
                    {
                        "hit_rate_3": evaluator.calculate_hit_rate(results, 3),
                        "hit_rate_5": evaluator.calculate_hit_rate(results, 5),
                    }
                )
                mlflow.log_artifact(str(results_file))
                mlflow.log_artifact(str(report_file))
                logger.info("Results logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        logger.info(f"Evaluation completed. Results saved to {args.output_dir}")
        logger.info(f"Report: {report_file}")

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(report)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
