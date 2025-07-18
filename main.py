from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import importlib
import pandas as pd
import csv
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sesuaikan dengan port FE-mu
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)


@app.post("/process")
async def process_file(
    code: str = Form(...),
    file: UploadFile = File(...)
):
    # Validasi ekstensi
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berupa CSV.")
    
    code = code.strip().lower()
    
    # Baca file CSV ke pandas
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file CSV: {str(e)}")
    
    required_columns = {
    "bulk_full_ip": [
        "user_id",
        "registration_ip",
        "event_timestamp",
        "ip_count",
        "user_name",
        "is_user_name_system_generated",
        "is_mitra",
        "registration_platform",
        "app_version",
        "registration_method",
        "dfpinfosz__securitydeviceid",
        "dfpinfosz__bssid",
        "grass_date",
        "count szdf",
        "grouping",
        "hit",
        "to check",
        "is_email_verified",
        "is_seller_ordered",
        "sum_buyer_mp_placed_order_cnt_1d",
        "checker",
        "action"
    ],

    "bulk_prefix_ip": [
        "user_id",
        "prefix",
        "registration_ip",
        "event_timestamp",
        "hour",
        "users",
        "user_name",
        "is_user_name_system_generated",
        "is_mitra",
        "registration_platform",
        "app_version",
        "registration_method",
        "grass_date",
        "status",
        "action",
        "grouping",
        "hit",
        "checker",
        "is_email_verified",
        "is_seller_ordered",
        "sum_buyer_mp_placed_order_cnt_1d"
    ],

    "bulk_mp1": [
        "user_id",
        "user_name",
        "cluster_size",
        "is_user_name_system_generated",
        "registration_datetime",
        "registration_ip",
        "ip_country",
        "registration_channel",
        "registration_platform",
        "user_registration_sz_did",
        "is_email_verified",
        "email",
        "is_phone_verified",
        "phone_number",
        "app_version",
        "is_new_device_login_otp_disabled",
        "order_bought_cnt_td",
        "fraud_tag",
        "source",
        "cluster_id",
        "cluser_user_id",
        "report_date",
        "action_on_buyer",
        "action_on_seller",
        "action_on_order",
        "action_on_device",
        "action_on_ba",
        "action_on_coins",
        "agent",
        "is_fraud",
        "grouping",
        "hit",
        "email verif",
        "seller order",
        "buyer order"
    ]
    }

    # Cek apakah kolom yang diperlukan ada di dalam file
    if code not in required_columns:
        raise HTTPException(status_code=400, detail=f"Processor '{code}' tidak dikenali.")
    
    missing_columns = [col for col in required_columns[code] if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"File tidak memiliki kolom yang diperlukan: {', '.join(missing_columns)}")

    
    # Dinamis load processor dari folder 'processors'
    try:
        processor = importlib.import_module(f"processors.{code}")
        processed_df = processor.process(df)
    except ModuleNotFoundError:
        raise HTTPException(status_code=400, detail=f"Processor '{code}' tidak ditemukan.")
    except AttributeError:
        raise HTTPException(status_code=500, detail=f"Processor '{code}' tidak memiliki fungsi 'process'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat memproses: {str(e)}")
    
    # Konversi hasil kembali ke CSV
    buffer = io.StringIO()
    processed_df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Format nama file hasil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{code}_{timestamp}.csv"

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'}
    )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # railway kasih PORT lewat env
    uvicorn.run("main:app", host="0.0.0.0", port=port)
