from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
import json

from .services import (
    CLASSES,
    run_prediction,
    run_gemini_analysis,
    ensure_model_loaded,
    ensure_gemini_loaded,
    get_health_status,
)
from .pdf import render_pdf


@require_GET
def index(request):
    ensure_model_loaded()
    ensure_gemini_loaded()
    return render(request, "diagnosis/index.html", {"classes": CLASSES})


@csrf_exempt
@require_POST
def predict_api(request):
    file = request.FILES.get("file")
    if not file:
        return HttpResponseBadRequest("File is required")
    if not file.content_type.startswith("image/"):
        return HttpResponseBadRequest("File must be an image")
    data = run_prediction(file.read())
    return JsonResponse(data)


@csrf_exempt
@require_POST
def analyze_api(request):
    file = request.FILES.get("file")
    if not file:
        return HttpResponseBadRequest("File is required")
    if not file.content_type.startswith("image/"):
        return HttpResponseBadRequest("File must be an image")
    data = run_gemini_analysis(file.read())
    return JsonResponse(data)


@require_GET
def health(request):
    return JsonResponse(get_health_status())


@require_GET
def classes(request):
    return JsonResponse({"classes": CLASSES, "count": len(CLASSES)})


@csrf_exempt
@require_POST
def predict_form(request):
    file = request.FILES.get("file")
    if not file:
        return HttpResponseBadRequest("File is required")
    if not file.content_type.startswith("image/"):
        return HttpResponseBadRequest("File must be an image")
    data = run_prediction(file.read())
    return render(
        request,
        "diagnosis/results.html",
        {"result": data, "result_json": json.dumps(data)},
    )


@csrf_exempt
@require_POST
def analyze_form(request):
    file = request.FILES.get("file")
    if not file:
        return HttpResponseBadRequest("File is required")
    if not file.content_type.startswith("image/"):
        return HttpResponseBadRequest("File must be an image")
    data = run_gemini_analysis(file.read())
    return render(
        request,
        "diagnosis/analysis.html",
        {"analysis": data, "analysis_json": json.dumps(data)},
    )


@csrf_exempt
@require_POST
def predict_pdf(request):
    payload = request.POST.get("data")
    if not payload:
        return HttpResponseBadRequest("Missing data")
    data = json.loads(payload)
    filename = f"skinwise_prediction_{data.get('request_id', '')}.pdf"
    return render_pdf("diagnosis/pdf_results.html", {"result": data}, filename)


@csrf_exempt
@require_POST
def analyze_pdf(request):
    payload = request.POST.get("data")
    if not payload:
        return HttpResponseBadRequest("Missing data")
    data = json.loads(payload)
    filename = f"skinwise_analysis_{data.get('request_id', '')}.pdf"
    return render_pdf("diagnosis/pdf_analysis.html", {"analysis": data}, filename)


# Create your views here.
