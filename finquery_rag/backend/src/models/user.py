from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from ..database import Base

class User(Base):
    """
    用户模型类，映射数据库中的 'users' 表。
    用于存储用户的基本信息，包括唯一标识、邮箱、加密密码以及创建时间。
    继承自 Base 类，利用 SQLAlchemy 的 ORM 机制进行数据库操作。
    """

    __tablename__ = "users"
    # 指定该模型在数据库中对应的表名为 'users'

    id = Column(Integer, primary_key=True, index=True)
    # 用户唯一标识，整数类型，作为主键并创建索引

    email = Column(String, unique=True, index=True)
    # 用户邮箱，字符串类型，值必须唯一并创建索引

    hashed_password = Column(String)
    # 用户的加密密码，字符串类型

    created_at = Column(DateTime, default=datetime.utcnow)
    # 用户记录的创建时间，日期时间类型，默认值为当前 UTC 时间