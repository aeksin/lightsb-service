import copy
import logging
import os
import shutil
import tempfile
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.auth.hashing import Hash
from app.auth.utils import create_jwt_token, get_user_by_token, get_user_id_from_token
from app.database.database import get_db
from app.database.models import UserDB
from app.database.user_db import user_repo
from app.database.user_profile_db import user_profile_repo
from app.database.user_stat_db import user_stats_repo
from app.exceptions import (
    DuplicateUserError,
    IncorrectPasswordError,
    NoGeneratedImagesError,
    NoRefreshTokenError,
    PasswordsDoNotMatchError,
    UserNotFoundError,
    UserProfileNotFoundError,
    UserProfileUpdateError,
    UserStatsNotFoundError,
    UserStatsUpdateError,
)
from app.resources.schemas import (
    User,
    UserBase,
    UserLogin,
    UserProfile,
    UserProfileUpdate,
    UserRegister,
    UserStats,
    UserStatsUpdate,
)
from models.aging_pipeline import aging_pipeline

router = APIRouter(tags=["auth"])


# Регистрация пользователя
@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register_user(request: UserRegister, db: Session = Depends(get_db)):
    if request.password != request.password_confirm:
        raise PasswordsDoNotMatchError()

    if user_repo.get_user_by_username(db, request.username):
        raise DuplicateUserError(username=request.username)

    if user_repo.get_user_by_email(db, request.email):
        raise DuplicateUserError(email=request.email)

    request.password = Hash.hash_password(request.password)

    new_user = user_repo.create_user(db, request)

    return JSONResponse(
        content={"redirect_to": "profile", "user_id": new_user.user_id}, status_code=201
    )


# Вход пользователя
@router.post("/auth/login", status_code=status.HTTP_200_OK)
def login_user(request: UserLogin, response: Response, db: Session = Depends(get_db)):
    if "@" in request.identifier:
        user = user_repo.get_user_by_email(db, request.identifier)
        email = request.identifier
        username = None
    else:
        user = user_repo.get_user_by_username(db, request.identifier)
        email = None
        username = request.identifier

    if not user:
        raise UserNotFoundError(email=email, username=username)

    if not Hash.verify_password(request.password, user.password):
        raise IncorrectPasswordError()

    refresh_expiry = timedelta(days=30) if request.remember_me else timedelta(days=7)

    access_token = create_jwt_token({"sub": user.user_id}, expires_delta=None)
    refresh_token = create_jwt_token(
        {"sub": user.user_id}, expires_delta=refresh_expiry
    )

    response.set_cookie(
        key="refresh_token_lightsb",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="None",
        max_age=30 * 24 * 60 * 60 if request.remember_me else None,
    )

    return JSONResponse(
        content={
            "access_token_lightsb": access_token,
            "redirect_to": "profile",
        },
        status_code=200,
        headers=response.headers,
    )


# Обновление токена
@router.post("/auth/refresh", status_code=status.HTTP_200_OK)
def refresh_token(request: Request, db: Session = Depends(get_db)):
    refresh_token = request.cookies.get("refresh_token_lightsb")

    if not refresh_token:
        raise NoRefreshTokenError()

    user_id = get_user_id_from_token(refresh_token)

    new_access_token = create_jwt_token({"sub": user_id}, expires_delta=None)

    return JSONResponse(
        content={"access_token_lightsb": new_access_token},
        status_code=200,
    )


# Обновление профиля
@router.post("/profile/update", status_code=status.HTTP_200_OK)
def update_profile(
    request: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_user_by_token),
):
    profile = user_profile_repo.get_user_profile(db, current_user.user_id)

    if not profile:
        raise UserProfileNotFoundError()

    updated_profile = user_profile_repo.update_user_profile(db, profile, request)

    if not updated_profile:
        raise UserProfileUpdateError()

    return JSONResponse(
        content={"redirect_to": "profile"},
        status_code=200,
    )


@router.get("/protected-route")
async def protected_route(current_user: UserDB = Depends(get_user_by_token)):
    return {"message": "Вы авторизованы!", "user": current_user.user_id}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("refresh_token_lightsb")
    return {"message": "Logged out"}


@router.get("/profile", response_model=UserProfile)
def get_profile(
    db: Session = Depends(get_db), current_user: UserDB = Depends(get_user_by_token)
):
    if not current_user:
        raise UserNotFoundError()

    profile = user_profile_repo.get_user_profile(db, current_user.user_id)

    if not profile:
        raise UserProfileNotFoundError()

    return {
        "profile_id": profile.profile_id,
        "user_id": profile.user_id,
        "full_name": profile.full_name,
        "position": profile.position,
        "date_of_birth": profile.date_of_birth,
    }


@router.get("/user", response_model=User)
def get_user(
    db: Session = Depends(get_db), current_user: UserDB = Depends(get_user_by_token)
):

    if not current_user:
        raise UserNotFoundError()

    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "user_id": current_user.user_id,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at,
        "is_active": current_user.is_active,
    }


@router.get("/stats", response_model=UserStats)
def get_stats(
    db: Session = Depends(get_db), current_user: UserDB = Depends(get_user_by_token)
):

    if not current_user:
        raise UserNotFoundError()

    stats = user_stats_repo.get_user_stats(db, current_user.user_id)

    if not stats:
        raise UserStatsNotFoundError()

    return {
        "stat_id": stats.stat_id,
        "user_id": stats.user_id,
        "usage_count": stats.usage_count,
        "images_count": stats.images_count,
        "last_used": stats.last_used,
        "avg_usage_time": stats.avg_usage_time,
    }


# Обновление статистики
@router.post("/stats/update", status_code=status.HTTP_200_OK)
def update_stats(
    request: UserStatsUpdate,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_user_by_token),
):
    stats = user_stats_repo.get_user_stats(db, current_user.user_id)

    if not stats:
        raise UserStatsNotFoundError()

    stats_update = copy.copy(stats)

    if not stats.avg_usage_time:
        stats_update.avg_usage_time = request.avg_usage_time
    else:
        stats_update.avg_usage_time = (
            stats.avg_usage_time * stats.usage_count
            + Decimal(str(request.avg_usage_time))
        ) / (stats.usage_count + request.usage_count)
    stats_update.usage_count += request.usage_count
    stats_update.images_count += request.images_count
    stats_update.last_used = request.last_used
    updated_stats = user_stats_repo.update_user_stats(db, stats, stats_update)

    if not updated_stats:
        raise UserStatsUpdateError()

    return JSONResponse(
        content={},
        status_code=200,
    )


@router.post("/generate", response_model=UserStats)
async def generate_images(file: UploadFile = File(...), model: str = Form(...)):
    try:
        folder_path_generated = "app/static/generated"

        shutil.rmtree(folder_path_generated, ignore_errors=True)
        os.makedirs(folder_path_generated, exist_ok=True)

        original_extension = Path(file.filename).suffix.lower() or ".jpg"
        input_filename = f"input_img{original_extension}"

        input_path = os.path.join(folder_path_generated, input_filename)

        image_data = await file.read()
        with open(input_path, "wb") as f:
            f.write(image_data)

        output_dir = os.path.join(folder_path_generated)
        generated_images = aging_pipeline(input_path, output_dir, model)
        if not generated_images:
            raise NoGeneratedImagesError
        elif len(generated_images) == 0:
            return JSONResponse(content={"image_urls": generated_images})

        clean_paths = [path.replace("app/", "", 1) for path in generated_images]

        return JSONResponse(content={"image_urls": clean_paths})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
