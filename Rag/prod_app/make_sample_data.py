"""Generates sample PDFs (HR policy docs) for the production app WITHOUT extra deps.

Usage:  python make_sample_data.py
Writes: data/docs/leave_policy.pdf  data/docs/salary_policy.pdf
"""

import pathlib

POLICIES = [
    ("leave_policy.pdf", [
        "Acme Corp Leave Policy 2026",
        "Earned Leave",
        "Employees earn 24 days of earned leave per year, credited at the "
        "start of each calendar year. Unused leave can be carried forward up "
        "to 10 days into the next year.",
        "Sick Leave",
        "Employees are entitled to 12 days of sick leave per year, available "
        "from the first day of joining. Sick leave requires a medical "
        "certificate for absences longer than two consecutive days.",
        "Special Leave",
        "Maternity leave is 26 weeks. Paternity leave is 2 weeks. Bereavement "
        "leave is 3 days per occurrence.",
    ]),
    ("salary_policy.pdf", [
        "Acme Corp Compensation Policy 2026",
        "Salary Review",
        "Salary reviews happen every April. Appraisals determine a 0-15% "
        "increment range based on performance banding.",
        "Bonuses",
        "An annual performance bonus of up to 20% of base salary is paid in "
        "May, tied to individual and company performance.",
        "Overtime",
        "Non-exempt employees receive 1.5x hourly rate for hours above 44 per "
        "week. Exempt employees do not receive overtime pay.",
    ]),
]


def _build_pdf(title: str, paragraphs: list[str]) -> bytes:
    """Hand-construct a minimal 1-page PDF with a single Helvetica font."""
    body_lines = [title, " "]
    for para in paragraphs:
        body_lines.extend(para.splitlines())
        body_lines.append(" ")

    stream_lines = []
    stream_lines.append("BT /F1 12 Tf 72 720 Td 14 TL")
    for line in body_lines:
        stream_lines.append(f"({line}) Tj T*")
    stream_lines.append("ET")
    content = "\n".join(stream_lines)

    stream_data = f"stream\n{content}\nendstream".encode("latin-1")
    stream_len = len(content.encode("latin-1"))

    objects = []
    objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    objects.append((2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    objects.append((3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"))
    objects.append((4, b"<< /Length " + str(stream_len).encode("latin-1") +
                       b" >>\n" + stream_data))
    objects.append((5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for num, body in objects:
        offsets.append(len(out))
        out.extend(f"{num} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return bytes(out)


def main() -> None:
    out_dir = pathlib.Path("data/docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, paragraphs in POLICIES:
        pdf = _build_pdf(name.replace(".pdf", "").replace("_", " ").title(), paragraphs)
        (out_dir / name).write_bytes(pdf)
        print(f"wrote {out_dir / name} ({len(pdf)} bytes)")


if __name__ == "__main__":
    main()