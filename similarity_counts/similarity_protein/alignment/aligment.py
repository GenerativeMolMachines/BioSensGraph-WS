from Bio import pairwise2


def calculate_alignment_similarity(sequence1, sequence2, alignment_type="global", match_score=2, mismatch_penalty=-1, gap_penalty=-0.5, extension_penalty=-0.1):
 """
 Calculates percent identity from pairwise alignment of two sequences (DNA or protein).

 Args:
 sequence1 (str): Pervaya sequence (DNK or belok).
 sequence2 (str): Vtoraya sequence.
 alignment_type (str): Alignment type: "global" (algorithm Needleman-Wunsch) or "local" (algorithm Smith-Waterman). Default value: "global".
 match_score (int): Score for match characters. Default value: 2.
 mismatch_penalty (int): Penalty for nematch characters. Default value: -1.
 gap_penalty (float): Penalty for gep (propusk). Default value: -0.5.
 extension_penalty (float): Penalty for rasshirenie gepa. Default value: -0.1.

 Returns:
 float: Percent sovpavshikh simvoloin in two string alignments (between 0 and 100). Returns 0 if alignment fails.
 """

    if alignment_type not in ["global", "local"]:
        raise ValueError("alignment_type must be 'global' or 'local'")

    if not sequence1 or not sequence2:
        return 0.0

    try:
        if alignment_type == "global":
            alignments = pairwise2.align.globalms(sequence1, sequence2, match_score, mismatch_penalty, gap_penalty, extension_penalty)
        else:
            alignments = pairwise2.align.localms(sequence1, sequence2, match_score, mismatch_penalty, gap_penalty, extension_penalty)
    except Exception as e:
        print(f"Alignment error: {e}")
        return 0.0

    if not alignments:
        return 0.0

    alignment1, alignment2, score, begin, end = alignments[0]

    matches = sum(1 for i in range(len(alignment1)) if alignment1[i] == alignment2[i] and alignment1[i] != '-')

    alignment_length = len(alignment1)

    similarity_percentage = (float(matches) / alignment_length) * 100 if alignment_length > 0 else 0.0

    return similarity_percentage