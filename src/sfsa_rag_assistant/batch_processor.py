# src/sfsa_rag_assistant/batch_processor.py
"""
Batch processing for running multiple queries through the agentic RAG pipeline.

Reads questions from a CSV file and outputs detailed results including all
intermediate agent decisions and responses.
"""

import csv
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm

from .graph import run_workflow

logger = logging.getLogger(__name__)


def process_batch(
    input_csv: str,
    output_csv: str,
    question_column: Optional[str] = None,
    max_validation_attempts: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Process a batch of questions from CSV through the agentic RAG pipeline.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file containing questions
    question_column : Optional[str]
        Name of the column containing questions. If omitted, auto-detect.
    output_csv : str
        Path to output CSV file for results
    max_validation_attempts : Optional[int]
        Maximum validation attempts (default: from settings)
    progress_callback : Optional[Callable[[int, int, str], None]]
        Callback invoked as rows are processed with `(completed, total, status)`
    
    Returns
    -------
    Dict[str, Any]
        Summary statistics about the batch processing
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    # Read input CSV
    logger.info(f"Reading questions from: {input_csv}")
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"No data found in {input_csv}")
    
    question_column = _detect_question_column(rows, question_column)
    
    logger.info(f"Found {len(rows)} questions to process")
    logger.info(f"Using question column: {question_column}")
    
    # Process each question
    results = []
    stats = {
        "total": len(rows),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "question_column": question_column,
        "agent1_sufficient": 0,
        "agent1_insufficient": 0,
        "agent3_first_pass": 0,
        "agent3_second_pass": 0,
        "agent3_failed": 0
    }

    if progress_callback:
        progress_callback(0, len(rows), f"Starting batch run for {len(rows)} questions")
    
    for i, row in enumerate(tqdm(rows, desc="Processing questions"), 1):
        question = row.get(question_column, "").strip()
        
        if not question:
            logger.warning(f"Row {i}: Empty question, skipping")
            stats["skipped"] += 1
            stats["processed"] += 1
            if progress_callback:
                progress_callback(stats["processed"], len(rows), f"Skipped empty row {i} of {len(rows)}")
            continue
        
        logger.info(f"Processing question {i}/{len(rows)}: {question[:50]}...")
        
        try:
            # Run workflow
            result = run_workflow(
                user_query=question,
                max_validation_attempts=max_validation_attempts
            )
            
            # Extract detailed information
            detailed_result = _extract_detailed_results(row, result)
            results.append(detailed_result)
            
            # Update statistics
            stats["successful"] += 1
            _update_stats(stats, result)
            stats["processed"] += 1
            if progress_callback:
                progress_callback(
                    stats["processed"],
                    len(rows),
                    f"Answered question {stats['processed']} of {len(rows)}",
                )
            
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}", exc_info=True)
            stats["failed"] += 1
            
            # Add error row
            error_result = {**row}
            error_result.update({
                "Agent 1: If context sufficient": "ERROR",
                "Agent 1 reasoning": str(e),
                "Agent 2: Web search query": "",
                "Answer before Agent 3": "",
                "Agent 3 run 1: Is answer satisfactory?": "",
                "Agent 3 run 1: clarifying question": "",
                "Answer before Agent 3 2nd run": "",
                "Agent 3 run 2: Is answer satisfactory?": "",
                "Agent 3 run 2: clarifying question": "",
                "Final answer": f"ERROR: {str(e)}",
                "Sources": "",
                "User notes": ""
            })
            results.append(error_result)
            stats["processed"] += 1
            if progress_callback:
                progress_callback(
                    stats["processed"],
                    len(rows),
                    f"Question {stats['processed']} of {len(rows)} failed",
                )
    
    # Write output CSV
    logger.info(f"Writing results to: {output_csv}")
    _write_output_csv(results, output_path, question_column)
    
    # Log summary
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total questions: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Agent 1 - Context sufficient: {stats['agent1_sufficient']}")
    logger.info(f"Agent 1 - Context insufficient (web search): {stats['agent1_insufficient']}")
    logger.info(f"Agent 3 - Passed first validation: {stats['agent3_first_pass']}")
    logger.info(f"Agent 3 - Needed second validation: {stats['agent3_second_pass']}")
    logger.info(f"Agent 3 - Failed after max attempts: {stats['agent3_failed']}")
    logger.info("=" * 80)

    if progress_callback:
        progress_callback(stats["processed"], len(rows), "Batch processing complete")
    
    return stats


def _format_sources_for_csv(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources list into a readable string for CSV output.
    
    Parameters
    ----------
    sources : List[Dict[str, Any]]
        List of source metadata dicts
    
    Returns
    -------
    str
        Formatted sources string
    """
    if not sources:
        return "No sources"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        label = _get_source_label(source, i)
        url = _get_source_url(source)
        formatted.append(f"{label}: {url}" if url else label)
    
    return " | ".join(formatted)


def _extract_detailed_results(original_row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract detailed agent decisions and responses from workflow result.
    
    Parameters
    ----------
    original_row : Dict[str, Any]
        Original row from input CSV
    result : Dict[str, Any]
        Result from run_workflow()
    
    Returns
    -------
    Dict[str, Any]
        Row with original data plus 11 new result columns (including sources)
    """
    metadata = result.get("metadata", {})
    
    # Start with original row data
    detailed = {**original_row}
    
    # Agent 1: Context Sufficiency
    context_info = metadata.get("context_sufficiency", {})
    detailed["Agent 1: If context sufficient"] = "Yes" if context_info.get("sufficient") else "No"
    detailed["Agent 1 reasoning"] = context_info.get("reasoning", "")
    
    # Agent 2: Web Search
    web_info = metadata.get("web_search", {})
    if web_info.get("used"):
        detailed["Agent 2: Web search query"] = web_info.get("query", "")
    else:
        detailed["Agent 2: Web search query"] = ""
    
    # Agent 3: Validation
    validation_info = metadata.get("validation", {})
    attempts = validation_info.get("attempts", 0)
    history = validation_info.get("history", [])
    
    # Answer before Agent 3 (first generation)
    if history and len(history) > 0:
        detailed["Answer before Agent 3"] = history[0].get("response", "")
        detailed["Agent 3 run 1: Is answer satisfactory?"] = "Yes" if history[0].get("satisfactory") else "No"
        detailed["Agent 3 run 1: clarifying question"] = "" if history[0].get("satisfactory") else history[0].get("refined_query", "")
    else:
        detailed["Answer before Agent 3"] = ""
        detailed["Agent 3 run 1: Is answer satisfactory?"] = ""
        detailed["Agent 3 run 1: clarifying question"] = ""
    
    # Second validation attempt (if it happened)
    if len(history) > 1:
        detailed["Answer before Agent 3 2nd run"] = history[1].get("response", "")
        detailed["Agent 3 run 2: Is answer satisfactory?"] = "Yes" if history[1].get("satisfactory") else "No"
        detailed["Agent 3 run 2: clarifying question"] = "" if history[1].get("satisfactory") else history[1].get("refined_query", "")
    else:
        detailed["Answer before Agent 3 2nd run"] = ""
        detailed["Agent 3 run 2: Is answer satisfactory?"] = ""
        detailed["Agent 3 run 2: clarifying question"] = ""
    
    # Sources - format as numbered list
    sources = result.get("sources", [])
    sources_formatted = _format_sources_for_csv(sources)
    detailed["Sources"] = sources_formatted
    detailed["_citation_links"] = _build_citation_link_cells(sources)
    
    # Final answer (always populated)
    detailed["Final answer"] = result.get("response", "")
    detailed["User notes"] = ""
    
    return detailed


def _update_stats(stats: Dict[str, Any], result: Dict[str, Any]):
    """Update statistics based on workflow result."""
    metadata = result.get("metadata", {})
    
    # Agent 1
    if metadata.get("context_sufficiency", {}).get("sufficient"):
        stats["agent1_sufficient"] += 1
    else:
        stats["agent1_insufficient"] += 1
    
    # Agent 3
    validation_info = metadata.get("validation", {})
    attempts = validation_info.get("attempts", 0)
    final_status = validation_info.get("final_status", "")
    
    if attempts == 1 and final_status == "satisfactory":
        stats["agent3_first_pass"] += 1
    elif attempts == 2 and final_status == "satisfactory":
        stats["agent3_second_pass"] += 1
    elif final_status == "needs_refinement":
        stats["agent3_failed"] += 1


def _write_output_csv(results: List[Dict[str, Any]], output_path: Path, question_column: str):
    """
    Write results to output CSV.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of result dictionaries
    output_path : Path
        Path to output CSV file
    question_column : str
        Name of the question column
    """
    if not results:
        logger.warning("No results to write")
        return
    
    # Get all original columns
    original_columns = [k for k in results[0].keys() if k not in [
        "Agent 1: If context sufficient",
        "Agent 1 reasoning",
        "Agent 2: Web search query",
        "Answer before Agent 3",
        "Agent 3 run 1: Is answer satisfactory?",
        "Agent 3 run 1: clarifying question",
        "Answer before Agent 3 2nd run",
        "Agent 3 run 2: Is answer satisfactory?",
        "Agent 3 run 2: clarifying question",
        "Final answer",
        "Sources",
        "User notes",
        "_citation_links"
    ]]

    max_citations = max(len(row.get("_citation_links", [])) for row in results)
    citation_columns = [f"Citation {index} link" for index in range(1, max_citations + 1)]

    for row in results:
        for index, column in enumerate(citation_columns, start=1):
            links = row.get("_citation_links", [])
            row[column] = links[index - 1] if index - 1 < len(links) else ""
        row.pop("_citation_links", None)
    
    # Define output column order
    output_columns = original_columns + [
        "Agent 1: If context sufficient",
        "Agent 1 reasoning",
        "Agent 2: Web search query",
        "Answer before Agent 3",
        "Agent 3 run 1: Is answer satisfactory?",
        "Agent 3 run 1: clarifying question",
        "Answer before Agent 3 2nd run",
        "Agent 3 run 2: Is answer satisfactory?",
        "Agent 3 run 2: clarifying question",
        "Final answer",
        "Sources",
        *citation_columns,
        "User notes"
    ]
    
    # Write CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Wrote {len(results)} rows to {output_path}")


def _detect_question_column(rows: List[Dict[str, Any]], requested_column: Optional[str]) -> str:
    """
    Detect the question column automatically when no explicit column is provided.
    """
    headers = list(rows[0].keys())

    if requested_column:
        if requested_column in headers:
            return requested_column
        raise ValueError(f"Column '{requested_column}' not found. Available columns: {headers}")

    preferred = {"question", "questions", "query", "prompt", "user_question"}
    for header in headers:
        if header.strip().lower() in preferred:
            return header

    for header in headers:
        if any((row.get(header) or "").strip() for row in rows):
            return header

    raise ValueError("Could not detect a question column in the uploaded CSV.")


def _get_source_url(source: Dict[str, Any]) -> str:
    """
    Return the best clickable URL for a source.
    """
    return source.get("wiki_url") or source.get("url") or ""


def _get_source_label(source: Dict[str, Any], index: int) -> str:
    """
    Build a concise label for a source entry.
    """
    source_type = source.get("source_type", "unknown")
    if source_type == "vector_db":
        source_file = Path(str(source.get("source", "N/A")).replace("\\", "/")).name
        page = source.get("page", "N/A")
        return f"[{index}] SFSA Wiki {source_file} (page {page})"
    return f"[{index}] Web {source.get('title', 'N/A')}"


def append_clickable_citations(response: str, sources: List[Dict[str, Any]]) -> str:
    """
    Replace [1] style citations with raw URL citations that survive CSV export.
    """
    if not response or not sources:
        return response

    def replace_match(match: re.Match[str]) -> str:
        linked_parts = []
        for part in [piece.strip() for piece in match.group(1).split(",")]:
            if not part.isdigit():
                linked_parts.append(part)
                continue
            index = int(part) - 1
            if index < 0 or index >= len(sources):
                linked_parts.append(part)
                continue
            url = _get_source_url(sources[index])
            linked_parts.append(f"Source {part}: {url}" if url else f"Source {part}")
        return " (" + "; ".join(linked_parts) + ")"

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace_match, response)


def _build_citation_link_cells(sources: List[Dict[str, Any]]) -> List[str]:
    """
    Build Excel-compatible hyperlink formulas for per-citation columns.
    """
    link_cells: List[str] = []
    for index, source in enumerate(sources, start=1):
        url = _get_source_url(source)
        label = _get_source_label(source, index)
        if not url:
            link_cells.append(label)
            continue
        safe_url = url.replace('"', '""')
        safe_label = label.replace('"', '""')
        link_cells.append(f'=HYPERLINK("{safe_url}","{safe_label}")')
    return link_cells
