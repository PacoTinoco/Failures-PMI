from pydantic import BaseModel, EmailStr, field_validator
from datetime import date
from typing import Optional


# ============================================================
# AUTH
# ============================================================

class MagicLinkRequest(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_corporate_email(cls, v: str) -> str:
        v = v.strip().lower()
        # Validación de dominio se hace en el router usando config.allowed_email_domains
        if "@" not in v:
            raise ValueError("Email inválido")
        return v


class TokenVerifyRequest(BaseModel):
    access_token: str


# ============================================================
# REGISTROS SEMANALES
# ============================================================

class RegistroSemanalCreate(BaseModel):
    operador_id: str
    semana: date
    # Sustentabilidad
    bos_num: Optional[int] = None
    bos_eng: Optional[float] = None
    # Calidad
    qbos_num: Optional[int] = None
    qbos_eng: Optional[float] = None
    qflags_num: Optional[int] = None
    qi_pnc_num: Optional[int] = 0
    # Desempeño
    dh_encontrados: Optional[int] = None
    dh_reparados: Optional[int] = None
    curva_autonomia: Optional[float] = None
    contramedidas_defectos: Optional[float] = None
    ips_num: Optional[int] = None
    # Costo
    frr: Optional[float] = None
    dim_waste: Optional[float] = None
    sobrepeso: Optional[int] = None
    eventos_laika: Optional[int] = 0
    # Moral
    casos_estudio: Optional[int] = None
    qm_on_target: Optional[float] = None
    # Meta
    estatus: Optional[str] = "Activo"
    notas: Optional[str] = None


class RegistroSemanalUpdate(BaseModel):
    bos_num: Optional[int] = None
    bos_eng: Optional[float] = None
    qbos_num: Optional[int] = None
    qbos_eng: Optional[float] = None
    qflags_num: Optional[int] = None
    qi_pnc_num: Optional[int] = None
    dh_encontrados: Optional[int] = None
    dh_reparados: Optional[int] = None
    curva_autonomia: Optional[float] = None
    contramedidas_defectos: Optional[float] = None
    ips_num: Optional[int] = None
    frr: Optional[float] = None
    dim_waste: Optional[float] = None
    sobrepeso: Optional[int] = None
    eventos_laika: Optional[int] = None
    casos_estudio: Optional[int] = None
    qm_on_target: Optional[float] = None
    estatus: Optional[str] = None
    notas: Optional[str] = None


# ============================================================
# OPERADORES / LC
# ============================================================

class CedulaCreate(BaseModel):
    nombre: str
    turno: Optional[str] = None
    grupo: Optional[str] = None
    notas: Optional[str] = None


class CedulaUpdate(BaseModel):
    nombre: Optional[str] = None
    turno: Optional[str] = None
    grupo: Optional[str] = None
    notas: Optional[str] = None
    activa: Optional[bool] = None


class OperadorCreate(BaseModel):
    nombre: str
    lc_id: str
    cedula_id: str
    turno: Optional[str] = None
    maquina: Optional[str] = None


class OperadorUpdate(BaseModel):
    nombre: Optional[str] = None
    lc_id: Optional[str] = None
    turno: Optional[str] = None
    maquina: Optional[str] = None
    activo: Optional[bool] = None


class LCCreate(BaseModel):
    nombre: str
    cedula_id: str
    turno: Optional[str] = None
    grupo: Optional[str] = None
    email: Optional[str] = None


class LCUpdate(BaseModel):
    nombre: Optional[str] = None
    turno: Optional[str] = None
    grupo: Optional[str] = None
    email: Optional[str] = None
    activo: Optional[bool] = None


class LSCreate(BaseModel):
    nombre: str
    cedula_id: str
    rol: Optional[str] = None
    email: Optional[str] = None
    turno: Optional[str] = None


class LSUpdate(BaseModel):
    nombre: Optional[str] = None
    rol: Optional[str] = None
    email: Optional[str] = None
    turno: Optional[str] = None
    activo: Optional[bool] = None


# ============================================================
# USUARIOS
# ============================================================

class UsuarioCreate(BaseModel):
    email: str
    nombre: str
    rol: str
    cedula_id: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("rol")
    @classmethod
    def validate_rol(cls, v: str) -> str:
        allowed = {"line_lead", "process_lead", "maintenance_lead", "becario"}
        if v not in allowed:
            raise ValueError(f"Rol debe ser uno de: {allowed}")
        return v
