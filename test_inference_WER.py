"""
Run Whisper ASR over a generated dataset split and report Word Error Rate (WER).

Expected dataset layout:
dataset_root/
	Eval/
		clean/
		text/

Text filename format:
	text_fileid_<sceneid>_doa<angle>_spk<k>.txt

Mapped clean filename format:
	clean_fileid_<sceneid>_doa<angle>_spk<k>.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import whisper
from scipy.signal import resample_poly
from tqdm import tqdm


def normalize_text(text: str) -> str:
	"""
	Lightweight normalization for fairer WER comparison.
	"""
	text = text.lower().strip()
	text = text.translate(str.maketrans("", "", string.punctuation))
	text = re.sub(r"\s+", " ", text)
	return text


def edit_distance_words(ref_words: List[str], hyp_words: List[str]) -> int:
	"""
	Classic Levenshtein distance at word level.
	"""
	n = len(ref_words)
	m = len(hyp_words)

	dp = np.zeros((n + 1, m + 1), dtype=np.int32)
	dp[:, 0] = np.arange(n + 1)
	dp[0, :] = np.arange(m + 1)

	for i in range(1, n + 1):
		for j in range(1, m + 1):
			cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
			dp[i, j] = min(
				dp[i - 1, j] + 1,      # deletion
				dp[i, j - 1] + 1,      # insertion
				dp[i - 1, j - 1] + cost,  # substitution
			)
	return int(dp[n, m])


def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
	"""
	Returns (wer_value, edit_distance, ref_word_count).
	"""
	ref_words = normalize_text(ref).split()
	hyp_words = normalize_text(hyp).split()

	n_ref = len(ref_words)
	if n_ref == 0:
		# Convention: if both empty then 0, else 1.
		return (0.0 if len(hyp_words) == 0 else 1.0, len(hyp_words), 0)

	dist = edit_distance_words(ref_words, hyp_words)
	return dist / n_ref, dist, n_ref


def text_to_clean_path(text_path: Path, clean_root: Path) -> Path:
	"""
	Map text_fileid_*_doa*_spk*.txt -> clean_fileid_*_doa*_spk*.wav
	"""
	stem = text_path.stem  # text_fileid_xxx_doaYYY_spkK
	if not stem.startswith("text_fileid_"):
		raise ValueError(f"Unexpected text filename: {text_path.name}")

	clean_stem = stem.replace("text_fileid_", "clean_fileid_", 1)
	return clean_root / f"{clean_stem}.wav"


def load_mono_channel(audio_path: Path, channel: int) -> np.ndarray:
	wav, sr = sf.read(str(audio_path), always_2d=True)  # [T, C]
	if channel < 0 or channel >= wav.shape[1]:
		raise ValueError(
			f"Requested channel={channel}, but audio has {wav.shape[1]} channels: {audio_path}"
		)

	mono = wav[:, channel].astype(np.float32)

	if sr != 16000:
		gcd = np.gcd(sr, 16000)
		up = 16000 // gcd
		down = sr // gcd
		mono = resample_poly(mono, up, down).astype(np.float32)
	return mono


@dataclass
class SampleResult:
	text_file: str
	audio_file: str
	reference: str
	hypothesis: str
	wer: float
	edit_distance: int
	ref_words: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate Whisper WER on full dataset split.")
	parser.add_argument(
		"--dataset_root",
		type=str,
		default=r"D:\邵鹏远\UCL\博1\code\Whisper_ASR\data\dataset_3mic_6spk",
		help="Root folder containing split folders (e.g., Eval)",
	)
	parser.add_argument("--split", type=str, default="Eval", help="Dataset split name")
	parser.add_argument("--model", type=str, default="turbo", help="Whisper model name")
	parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
	parser.add_argument("--language", type=str, default="en", help="Whisper language hint")
	parser.add_argument("--channel", type=int, default=0, help="Channel index to read from wav (default 0)")
	parser.add_argument(
		"--max_items",
		type=int,
		default=0,
		help="Optional cap for quick tests; 0 means use all items",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default="wer_results",
		help="Directory to store CSV and JSON summary",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	split_root = Path(args.dataset_root) / args.split
	text_root = split_root / "text"
	clean_root = split_root / "clean"

	if not text_root.is_dir() or not clean_root.is_dir():
		raise FileNotFoundError(f"Expected text/ and clean/ under: {split_root}")

	text_files = sorted(text_root.glob("*.txt"))
	if args.max_items and args.max_items > 0:
		text_files = text_files[: args.max_items]

	if len(text_files) == 0:
		raise FileNotFoundError(f"No text files found in: {text_root}")

	print(f"Loading Whisper model={args.model} on device={args.device}...")
	model = whisper.load_model(args.model, device=args.device)

	sample_results: List[SampleResult] = []
	total_edits = 0
	total_ref_words = 0
	missing_audio = 0

	for idx, text_path in enumerate(tqdm(text_files, desc="Evaluating", unit="utt"), start=1):
		ref_text = text_path.read_text(encoding="utf-8").strip()
		clean_path = text_to_clean_path(text_path, clean_root)

		if not clean_path.is_file():
			missing_audio += 1
			print(f"[{idx}/{len(text_files)}] missing audio: {clean_path.name}")
			continue

		audio = load_mono_channel(clean_path, args.channel)
		result = model.transcribe(audio, language=args.language, fp16=False)
		hyp_text = result.get("text", "").strip()

		sample_wer, dist, ref_words = wer(ref_text, hyp_text)
		total_edits += dist
		total_ref_words += ref_words

		sample_results.append(
			SampleResult(
				text_file=text_path.name,
				audio_file=clean_path.name,
				reference=ref_text,
				hypothesis=hyp_text,
				wer=sample_wer,
				edit_distance=dist,
				ref_words=ref_words,
			)
		)

		# if idx % 25 == 0 or idx == len(text_files):
		# 	print(f"Processed {idx}/{len(text_files)}")

	corpus_wer = (total_edits / total_ref_words) if total_ref_words > 0 else 0.0
	avg_sample_wer = (
		float(np.mean([x.wer for x in sample_results])) if sample_results else 0.0
	)

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	csv_path = out_dir / f"{args.split}_whisper_{args.model}_details.csv"

	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"text_file",
				"audio_file",
				"wer",
				"edit_distance",
				"ref_words",
				"reference",
				"hypothesis",
			],
		)
		writer.writeheader()
		for row in sample_results:
			writer.writerow(asdict(row))


	print("\n===== WER SUMMARY =====")
	print(f"Whisper model: {args.model}, device: {args.device}")
	print(f"Split: {args.split}")
	print(f"Requested items: {len(text_files)}")
	print(f"Evaluated items: {len(sample_results)}")
	print(f"Missing audio pairs: {missing_audio}")
	print(f"Corpus WER: {corpus_wer:.4f}")
	print(f"Average sample WER: {avg_sample_wer:.4f}")
	print(f"Saved details CSV: {csv_path}")


if __name__ == "__main__":
	main()
