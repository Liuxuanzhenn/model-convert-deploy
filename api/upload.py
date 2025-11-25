"""文件上传API接口"""
import os
import shutil
import tempfile
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from services.files import ExtraFilesManager
from utils.path import PathManager
from utils.error import create_error_response, create_success_response, ErrorCode

logger = logging.getLogger(__name__)

upload_api_bp = Blueprint('upload_api', __name__)


@upload_api_bp.post("/upload-extra-files")
def upload_extra_files():
    """上传额外文件（ZIP格式）
    ---
    tags:
      - 文件上传
    summary: 上传ZIP格式的额外文件（校准数据、训练数据等）
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: ZIP文件
      - in: formData
        name: extra_dir
        type: string
        required: true
        description: 目标目录路径
    responses:
      200:
        description: 上传成功
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            message:
              type: string
              example: success
            data:
              type: object
              properties:
                extra_dir:
                  type: string
                recognized_files:
                  type: object
                file_count:
                  type: integer
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    try:
        if 'file' not in request.files:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "No file provided"
            )), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "Empty filename"
            )), 400
        
        if not file.filename.lower().endswith('.zip'):
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "Only zip files are supported"
            )), 400
        
        extra_dir = request.form.get('extra_dir')
        if not extra_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "extra_dir is required"
            )), 400
        
        try:
            extra_dir = PathManager.validate_extra_dir(extra_dir, create_if_not_exists=True)
        except ValueError as e:
            return jsonify(create_error_response(
                ErrorCode.PATH_INVALID,
                str(e)
            )), 400
        
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_zip_path = os.path.join(temp_dir, filename)
        
        try:
            file.save(temp_zip_path)
            
            manager = ExtraFilesManager(extra_dir)
            result = manager.extract_and_distribute(temp_zip_path)
            
            return jsonify(create_success_response({
                "extra_dir": extra_dir,
                "recognized_files": result,
                "file_count": sum(len(files) for files in result.values())
            }))
        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    except ValueError as e:
        return jsonify(create_error_response(
            ErrorCode.BAD_REQUEST,
            str(e)
        )), 400
    except Exception as e:
        logger.error(f"Error in upload_extra_files: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500

