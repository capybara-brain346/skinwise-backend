import io
from typing import Dict, Any

from django.http import HttpResponse
from django.template.loader import render_to_string


def render_pdf(
    template_name: str, context: Dict[str, Any], filename: str
) -> HttpResponse:
    html = render_to_string(template_name, context)
    try:
        from xhtml2pdf import pisa
    except Exception as e:
        response = HttpResponse(str(e), content_type="text/plain")
        response.status_code = 500
        return response

    result = io.BytesIO()
    pisa.CreatePDF(src=html, dest=result, encoding="utf-8")
    pdf = result.getvalue()
    result.close()

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f"attachment; filename={filename}"
    return response
