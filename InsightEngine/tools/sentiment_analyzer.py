"""
Multilingual sentiment analysis tool
Provides sentiment analysis for InsightEngine based on the WeiboMultilingualSentiment model
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re

try:
    import torch

    TORCH_AVAILABLE = True
    torch.classes.__path__ = []
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


# INFO: To skip sentiment analysis, manually set this switch to False
SENTIMENT_ANALYSIS_ENABLED = True


def _describe_missing_dependencies() -> str:
    missing = []
    if not TORCH_AVAILABLE:
        missing.append("PyTorch")
    if not TRANSFORMERS_AVAILABLE:
        missing.append("Transformers")
    return " / ".join(missing)


# Add the project root directory to the path to import WeiboMultilingualSentiment
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
weibo_sentiment_path = os.path.join(
    project_root, "SentimentAnalysisModel", "WeiboMultilingualSentiment"
)
sys.path.append(weibo_sentiment_path)


@dataclass
class SentimentResult:
    """Sentiment analysis result data class"""

    text: str
    sentiment_label: str
    confidence: float
    probability_distribution: Dict[str, float]
    success: bool = True
    error_message: Optional[str] = None
    analysis_performed: bool = True


@dataclass
class BatchSentimentResult:
    """Batch sentiment analysis result data class"""

    results: List[SentimentResult]
    total_processed: int
    success_count: int
    failed_count: int
    average_confidence: float
    analysis_performed: bool = True


class WeiboMultilingualSentimentAnalyzer:
    """
    Multilingual sentiment analyzer
    Wraps the WeiboMultilingualSentiment model and provides sentiment analysis functionality for AI Agents
    """

    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_initialized = False
        self.is_disabled = False
        self.disable_reason: Optional[str] = None

        # Sentiment label mapping (5-class classification)
        self.sentiment_map = {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Very Positive",
        }

        if not SENTIMENT_ANALYSIS_ENABLED:
            self.disable("Sentiment analysis is disabled in configuration.")
        elif not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            missing = _describe_missing_dependencies() or "Unknown dependency"
            self.disable(f"Missing dependency: {missing}. Sentiment analysis has been disabled.")

        if self.is_disabled:
            reason = self.disable_reason or "Sentiment analysis disabled."
            print(
                f"WeiboMultilingualSentimentAnalyzer initialized but disabled: {reason}"
            )
        else:
            print(
                "WeiboMultilingualSentimentAnalyzer created. Call initialize() to load the model"
            )

    def disable(self, reason: Optional[str] = None, drop_state: bool = False) -> None:
        """Disable sentiment analysis, optionally clearing loaded resources."""
        self.is_disabled = True
        self.disable_reason = reason or "Sentiment analysis disabled."
        if drop_state:
            self.model = None
            self.tokenizer = None
            self.device = None
            self.is_initialized = False

    def enable(self) -> bool:
        """Attempt to enable sentiment analysis; returns True if enabled."""
        if not SENTIMENT_ANALYSIS_ENABLED:
            self.disable("Sentiment analysis is disabled in configuration.")
            return False
        if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            missing = _describe_missing_dependencies() or "Unknown dependency"
            self.disable(f"Missing dependency: {missing}. Sentiment analysis has been disabled.")
            return False
        self.is_disabled = False
        self.disable_reason = None
        return True

    def _select_device(self):
        """Select the best available torch device."""
        if not TORCH_AVAILABLE:
            return None
        assert torch is not None
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if (
            mps_backend
            and getattr(mps_backend, "is_available", lambda: False)()
            and getattr(mps_backend, "is_built", lambda: False)()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    def initialize(self) -> bool:
        """
        Initialize model and tokenizer

        Returns:
            Whether initialization succeeded
        """
        if self.is_disabled:
            reason = self.disable_reason or "Sentiment analysis is disabled"
            print(f"Sentiment analysis is disabled. Skipping model loading: {reason}")
            return False

        if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
            missing = _describe_missing_dependencies() or "Unknown dependency"
            self.disable(f"Missing dependency: {missing}. Sentiment analysis has been disabled.", drop_state=True)
            print(f"Missing dependency: {missing}. Unable to load sentiment analysis model.")
            return False

        if self.is_initialized:
            print("Model is already initialized, no need to load it again")
            return True

        try:
            print("Loading multilingual sentiment analysis model...")
            assert AutoTokenizer is not None
            assert AutoModelForSequenceClassification is not None

            # Use multilingual sentiment analysis model
            model_name = "tabularisai/multilingual-sentiment-analysis"
            local_model_path = os.path.join(weibo_sentiment_path, "model")

            # Check whether the model already exists locally
            if os.path.exists(local_model_path):
                print("Loading model from local path...")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    local_model_path
                )
            else:
                print("First run detected, downloading model locally...")
                # Download and save locally
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )

                # Save locally
                os.makedirs(local_model_path, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
                print(f"Model saved to: {local_model_path}")

            # Set device
            device = self._select_device()
            if device is None:
                raise RuntimeError("No available compute device detected")

            self.device = device
            self.model.to(self.device)
            self.model.eval()
            self.is_initialized = True
            self.enable()

            device_type = getattr(self.device, "type", str(self.device))
            if device_type == "cuda":
                print("Detected available GPU, CUDA is prioritized for inference.")
            elif device_type == "mps":
                print("Detected Apple MPS device, MPS is used for inference.")
            else:
                print("No GPU detected, using CPU for inference.")

            print(f"Model loaded successfully! Using device: {self.device}")
            print("Supported languages: Chinese, English, Spanish, Arabic, Japanese, Korean, and 22 languages total")
            print("Sentiment levels: Very Negative, Negative, Neutral, Positive, Very Positive")

            return True

        except Exception as e:
            error_message = f"Model loading failed: {e}"
            print(error_message)
            print("Please check network connection or model files")
            self.disable(error_message, drop_state=True)
            return False

    def _preprocess_text(self, text: str) -> str:
        """
        Text preprocessing

        Args:
            text: Input text

        Returns:
            Processed text
        """
        # Basic text cleanup
        if not text or not text.strip():
            return ""

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text.strip())

        return text

    def analyze_single_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment for a single text

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object
        """
        if self.is_disabled:
            return SentimentResult(
                text=text,
                sentiment_label="Sentiment analysis not executed",
                confidence=0.0,
                probability_distribution={},
                success=False,
                error_message=self.disable_reason or "Sentiment analysis is disabled",
                analysis_performed=False,
            )

        if not self.is_initialized:
            return SentimentResult(
                text=text,
                sentiment_label="Not initialized",
                confidence=0.0,
                probability_distribution={},
                success=False,
                error_message="Model is not initialized, please call initialize() first",
                analysis_performed=False,
            )

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            if not processed_text:
                return SentimentResult(
                    text=text,
                    sentiment_label="Input error",
                    confidence=0.0,
                    probability_distribution={},
                    success=False,
                    error_message="Input text is empty or invalid",
                    analysis_performed=False,
                )
            assert self.tokenizer is not None
            # Tokenize and encode
            inputs = self.tokenizer(
                processed_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            assert torch is not None
            assert self.model is not None
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = int(torch.argmax(probabilities, dim=1).item())

            # Build result
            confidence = probabilities[0][prediction].item()
            label = self.sentiment_map[prediction]

            # Build probability distribution dictionary
            prob_dist = {}
            for label_name, prob in zip(self.sentiment_map.values(), probabilities[0]):
                prob_dist[label_name] = prob.item()

            return SentimentResult(
                text=text,
                sentiment_label=label,
                confidence=confidence,
                probability_distribution=prob_dist,
                success=True,
            )

        except Exception as e:
            return SentimentResult(
                text=text,
                sentiment_label="Analysis failed",
                confidence=0.0,
                probability_distribution={},
                success=False,
                error_message=f"Error during prediction: {str(e)}",
                analysis_performed=False,
            )

    def analyze_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> BatchSentimentResult:
        """
        Batch sentiment analysis

        Args:
            texts: List of texts
            show_progress: Whether to show progress

        Returns:
            BatchSentimentResult object
        """
        if not texts:
            return BatchSentimentResult(
                results=[],
                total_processed=0,
                success_count=0,
                failed_count=0,
                average_confidence=0.0,
                analysis_performed=not self.is_disabled and self.is_initialized,
            )

        if self.is_disabled or not self.is_initialized:
            passthrough_results = [
                SentimentResult(
                    text=text,
                    sentiment_label="Sentiment analysis not executed",
                    confidence=0.0,
                    probability_distribution={},
                    success=False,
                    error_message=self.disable_reason or "Sentiment analysis is unavailable",
                    analysis_performed=False,
                )
                for text in texts
            ]
            return BatchSentimentResult(
                results=passthrough_results,
                total_processed=len(texts),
                success_count=0,
                failed_count=len(texts),
                average_confidence=0.0,
                analysis_performed=False,
            )

        results = []
        success_count = 0
        total_confidence = 0.0

        for i, text in enumerate(texts):
            if show_progress and len(texts) > 1:
                print(f"Processing progress: {i + 1}/{len(texts)}")

            result = self.analyze_single_text(text)
            results.append(result)

            if result.success:
                success_count += 1
                total_confidence += result.confidence

        average_confidence = (
            total_confidence / success_count if success_count > 0 else 0.0
        )
        failed_count = len(texts) - success_count

        return BatchSentimentResult(
            results=results,
            total_processed=len(texts),
            success_count=success_count,
            failed_count=failed_count,
            average_confidence=average_confidence,
            analysis_performed=True,
        )

    def _build_passthrough_analysis(
        self,
        original_data: List[Dict[str, Any]],
        reason: str,
        texts: Optional[List[str]] = None,
        results: Optional[List[SentimentResult]] = None,
    ) -> Dict[str, Any]:
        """
        Build passthrough result when sentiment analysis is unavailable
        """
        total_items = len(texts) if texts is not None else len(original_data)
        response: Dict[str, Any] = {
            "sentiment_analysis": {
                "available": False,
                "reason": reason,
                "total_analyzed": 0,
                "success_rate": f"0/{total_items}",
                "average_confidence": 0.0,
                "sentiment_distribution": {},
                "high_confidence_results": [],
                "summary": f"Sentiment analysis not executed: {reason}",
                "original_texts": original_data,
            }
        }

        if texts is not None:
            response["sentiment_analysis"]["passthrough_texts"] = texts

        if results is not None:
            response["sentiment_analysis"]["results"] = [
                result.__dict__ if isinstance(result, SentimentResult) else result
                for result in results
            ]

        return response

    def analyze_query_results(
        self,
        query_results: List[Dict[str, Any]],
        text_field: str = "content",
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for query results
        Specifically used to analyze query results returned by MediaCrawlerDB

        Args:
            query_results: Query result list, each item contains text content
            text_field: Text field name, default is "content"
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary containing sentiment analysis results
        """
        if not query_results:
            return {
                "sentiment_analysis": {
                    "total_analyzed": 0,
                    "sentiment_distribution": {},
                    "high_confidence_results": [],
                    "summary": "No content to analyze",
                }
            }

        # Extract text content
        texts_to_analyze = []
        original_data = []

        for item in query_results:
            # Try multiple possible text fields
            text_content = ""
            for field in [text_field, "title_or_content", "content", "title", "text"]:
                if field in item and item[field]:
                    text_content = str(item[field])
                    break

            if text_content.strip():
                texts_to_analyze.append(text_content)
                original_data.append(item)

        if not texts_to_analyze:
            return {
                "sentiment_analysis": {
                    "total_analyzed": 0,
                    "sentiment_distribution": {},
                    "high_confidence_results": [],
                    "summary": "No analyzable text content found in query results",
                }
            }

        if self.is_disabled:
            return self._build_passthrough_analysis(
                original_data=original_data,
                reason=self.disable_reason or "Sentiment analysis model is unavailable",
                texts=texts_to_analyze,
            )

        # Run batch sentiment analysis
        print(f"Running sentiment analysis for {len(texts_to_analyze)} items...")
        batch_result = self.analyze_batch(texts_to_analyze, show_progress=True)

        if not batch_result.analysis_performed:
            reason = self.disable_reason or "Sentiment analysis is unavailable"
            if batch_result.results:
                candidate_error = next(
                    (r.error_message for r in batch_result.results if r.error_message),
                    None,
                )
                if candidate_error:
                    reason = candidate_error
            return self._build_passthrough_analysis(
                original_data=original_data,
                reason=reason,
                texts=texts_to_analyze,
                results=batch_result.results,
            )

        # Aggregate sentiment distribution
        sentiment_distribution = {}
        high_confidence_results = []

        for result, original_item in zip(batch_result.results, original_data):
            if result.success:
                # Aggregate sentiment distribution
                sentiment = result.sentiment_label
                if sentiment not in sentiment_distribution:
                    sentiment_distribution[sentiment] = 0
                sentiment_distribution[sentiment] += 1

                # Collect high-confidence results
                if result.confidence >= min_confidence:
                    high_confidence_results.append(
                        {
                            "original_data": original_item,
                            "sentiment": result.sentiment_label,
                            "confidence": result.confidence,
                            "text_preview": result.text[:100] + "..."
                            if len(result.text) > 100
                            else result.text,
                        }
                    )

        # Generate sentiment analysis summary
        total_analyzed = batch_result.success_count
        if total_analyzed > 0:
            dominant_sentiment = max(sentiment_distribution.items(), key=lambda x: x[1])
            sentiment_summary = f"Analyzed {total_analyzed} items; the dominant sentiment is '{dominant_sentiment[0]}' ({dominant_sentiment[1]} items, {dominant_sentiment[1] / total_analyzed * 100:.1f}%)."
        else:
            sentiment_summary = "Sentiment analysis failed"

        return {
            "sentiment_analysis": {
                "total_analyzed": total_analyzed,
                "success_rate": f"{batch_result.success_count}/{batch_result.total_processed}",
                "average_confidence": round(batch_result.average_confidence, 4),
                "sentiment_distribution": sentiment_distribution,
                "high_confidence_results": high_confidence_results,  # Return all high-confidence results without limiting count
                "summary": sentiment_summary,
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Model information dictionary
        """
        return {
            "model_name": "tabularisai/multilingual-sentiment-analysis",
            "supported_languages": [
                "Chinese",
                "English",
                "Spanish",
                "Arabic",
                "Japanese",
                "Korean",
                "German",
                "French",
                "Italian",
                "Portuguese",
                "Russian",
                "Dutch",
                "Polish",
                "Turkish",
                "Danish",
                "Greek",
                "Finnish",
                "Swedish",
                "Norwegian",
                "Hungarian",
                "Czech",
                "Bulgarian",
            ],
            "sentiment_levels": list(self.sentiment_map.values()),
            "is_initialized": self.is_initialized,
            "device": str(self.device) if self.device else "Not set",
        }


# Create global instance (lazy initialization)
multilingual_sentiment_analyzer = WeiboMultilingualSentimentAnalyzer()


def enable_sentiment_analysis() -> bool:
    """Public helper to enable sentiment analysis at runtime."""
    return multilingual_sentiment_analyzer.enable()


def disable_sentiment_analysis(
    reason: Optional[str] = None, drop_state: bool = False
) -> None:
    """Public helper to disable sentiment analysis at runtime."""
    multilingual_sentiment_analyzer.disable(reason=reason, drop_state=drop_state)


def analyze_sentiment(
    text_or_texts: Union[str, List[str]], initialize_if_needed: bool = True
) -> Union[SentimentResult, BatchSentimentResult]:
    """
    Convenience sentiment analysis function

    Args:
        text_or_texts: Single text or text list
        initialize_if_needed: Whether to auto-initialize when model is not initialized

    Returns:
        SentimentResult or BatchSentimentResult
    """
    if (
        initialize_if_needed
        and not multilingual_sentiment_analyzer.is_initialized
        and not multilingual_sentiment_analyzer.is_disabled
    ):
        multilingual_sentiment_analyzer.initialize()

    if isinstance(text_or_texts, str):
        return multilingual_sentiment_analyzer.analyze_single_text(text_or_texts)
    else:
        texts_list = list(text_or_texts)
        return multilingual_sentiment_analyzer.analyze_batch(texts_list)


if __name__ == "__main__":
    # Test code
    analyzer = WeiboMultilingualSentimentAnalyzer()

    if analyzer.initialize():
        # Test a single text
        result = analyzer.analyze_single_text("The weather is wonderful today, and I feel great!")
        print(
            f"Single-text analysis: {result.sentiment_label} (confidence: {result.confidence:.4f})"
        )

        # Test batch texts
        test_texts = [
            "The food at this restaurant tastes amazing!",
            "The service attitude is terrible, very disappointed.",
            "I absolutely love this product!",
            "The customer service was disappointing.",
        ]

        batch_result = analyzer.analyze_batch(test_texts)
        print(
            f"\nBatch analysis: success {batch_result.success_count}/{batch_result.total_processed}"
        )

        for result in batch_result.results:
            print(
                f"'{result.text[:30]}...' -> {result.sentiment_label} ({result.confidence:.4f})"
            )
    else:
        print("Model initialization failed, unable to run tests")
