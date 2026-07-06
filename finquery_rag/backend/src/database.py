from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# 从环境变量中获取数据库连接URL
DATABASE_URL = os.getenv("DATABASE_URL")

# 处理 DATABASE_URL 可能未设置的情况（例如在构建期间）
if not DATABASE_URL:
    # 如果未指定，则默认使用 sqlite 进行本地开发，或者也可以抛出错误
    DATABASE_URL = "sqlite:///./temp.db" 

# 创建数据库引擎实例，用于管理数据库连接池
engine = create_engine(DATABASE_URL)

# 创建数据库会话工厂类，配置为非自动提交和非自动刷新，并将引擎绑定到该会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建声明式基类，用于后续定义 ORM 模型
Base = declarative_base()

def get_db():
    """
    获取数据库会话的生成器函数。
    
    用于在依赖注入系统（如 FastAPI 的 Depends）中提供数据库会话，
    确保在请求处理完成后正确关闭数据库会话，释放连接资源。

    Yields:
        Session: SQLAlchemy 数据库会话实例，用于执行数据库操作。
    """
    # 创建一个新的数据库会话实例
    db = SessionLocal()
    try:
        # 将会话实例提供给调用方使用
        yield db
    finally:
        # 无论是否发生异常，最终都会关闭会话，确保资源被正确释放
        db.close()