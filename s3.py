import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import io


class S3Service:
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")

        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME must be provided or set in environment")

        session_kwargs = {}
        if region_name or os.getenv("AWS_REGION"):
            session_kwargs["region_name"] = region_name or os.getenv("AWS_REGION")
        if aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"):
            session_kwargs["aws_access_key_id"] = aws_access_key_id or os.getenv(
                "AWS_ACCESS_KEY_ID"
            )
        if aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"):
            session_kwargs["aws_secret_access_key"] = (
                aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            )

        self.s3_client = boto3.client("s3", **session_kwargs)
        self.s3_resource = boto3.resource("s3", **session_kwargs)

    def download_latest_model(
        self, model_prefix: str = "models/", local_path: Optional[Path] = None
    ) -> Path:
        try:
            bucket = self.s3_resource.Bucket(self.bucket_name)

            model_objects = list(bucket.objects.filter(Prefix=model_prefix))

            if not model_objects:
                raise FileNotFoundError(
                    f"No models found in bucket with prefix '{model_prefix}'"
                )

            onnx_models = [obj for obj in model_objects if obj.key.endswith(".onnx")]

            if not onnx_models:
                raise FileNotFoundError(
                    f"No ONNX models found in bucket with prefix '{model_prefix}'"
                )

            latest_model = max(onnx_models, key=lambda x: x.last_modified)

            if local_path is None:
                artifacts_dir = Path("artifacts")
                artifacts_dir.mkdir(exist_ok=True)
                local_path = artifacts_dir / Path(latest_model.key).name
            else:
                local_path = Path(local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)

            print(
                f"Downloading model from S3: {latest_model.key} (Last modified: {latest_model.last_modified})"
            )
            self.s3_client.download_file(
                self.bucket_name, latest_model.key, str(local_path)
            )
            print(f"Model downloaded successfully to: {local_path}")

            return local_path

        except NoCredentialsError:
            raise Exception(
                "AWS credentials not found. Please configure AWS credentials."
            )
        except ClientError as e:
            raise Exception(f"Error downloading model from S3: {str(e)}")

    def upload_image(
        self,
        image_data: bytes,
        request_id: str,
        image_filename: str = "input_image.jpg",
    ) -> str:
        try:
            s3_key = f"predictions/{request_id}/{image_filename}"

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_data,
                ContentType="image/jpeg",
            )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            print(f"Image uploaded successfully: {s3_url}")

            return s3_url

        except NoCredentialsError:
            raise Exception(
                "AWS credentials not found. Please configure AWS credentials."
            )
        except ClientError as e:
            raise Exception(f"Error uploading image to S3: {str(e)}")

    def upload_prediction(
        self,
        prediction_data: Dict[str, Any],
        request_id: str,
        filename: str = "prediction.json",
    ) -> str:
        try:
            s3_key = f"predictions/{request_id}/{filename}"

            json_data = json.dumps(prediction_data, indent=2)

            metadata = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "content_type": "application/json",
            }

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode("utf-8"),
                ContentType="application/json",
                Metadata=metadata,
            )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            print(f"Prediction data uploaded successfully: {s3_url}")

            return s3_url

        except NoCredentialsError:
            raise Exception(
                "AWS credentials not found. Please configure AWS credentials."
            )
        except ClientError as e:
            raise Exception(f"Error uploading prediction to S3: {str(e)}")

    def upload_analysis(
        self,
        analysis_data: Dict[str, Any],
        request_id: str,
        filename: str = "analysis.json",
    ) -> str:
        try:
            s3_key = f"predictions/{request_id}/{filename}"

            json_data = json.dumps(analysis_data, indent=2)

            metadata = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "content_type": "application/json",
            }

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode("utf-8"),
                ContentType="application/json",
                Metadata=metadata,
            )

            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            print(f"Analysis data uploaded successfully: {s3_url}")

            return s3_url

        except NoCredentialsError:
            raise Exception(
                "AWS credentials not found. Please configure AWS credentials."
            )
        except ClientError as e:
            raise Exception(f"Error uploading analysis to S3: {str(e)}")

    def get_prediction(
        self, request_id: str, filename: str = "prediction.json"
    ) -> Dict[str, Any]:
        try:
            s3_key = f"predictions/{request_id}/{filename}"

            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_data = response["Body"].read().decode("utf-8")

            return json.loads(json_data)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(
                    f"Prediction not found for request_id: {request_id}"
                )
            raise Exception(f"Error retrieving prediction from S3: {str(e)}")

    def list_predictions(self, prefix: str = "predictions/") -> list[str]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"
            )

            request_ids = []
            if "CommonPrefixes" in response:
                for prefix_info in response["CommonPrefixes"]:
                    request_id = prefix_info["Prefix"].rstrip("/").split("/")[-1]
                    request_ids.append(request_id)

            return request_ids

        except ClientError as e:
            raise Exception(f"Error listing predictions from S3: {str(e)}")

    def delete_prediction(self, request_id: str) -> bool:
        try:
            bucket = self.s3_resource.Bucket(self.bucket_name)
            objects_to_delete = bucket.objects.filter(
                Prefix=f"predictions/{request_id}/"
            )

            deleted_count = 0
            for obj in objects_to_delete:
                obj.delete()
                deleted_count += 1

            print(f"Deleted {deleted_count} objects for request_id: {request_id}")
            return True

        except ClientError as e:
            raise Exception(f"Error deleting prediction from S3: {str(e)}")


def get_s3_service() -> S3Service:
    return S3Service()
