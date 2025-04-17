from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.config import engine, SessionLocal
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# User Profile Model
class UserProfile(Base):
    __tablename__ = 'user_profiles'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    progress = relationship("TopicProgress", back_populates="user_profile")
    weaknesses = relationship("Weakness", back_populates="user_profile")

# Topic Progress Model
class TopicProgress(Base):
    __tablename__ = 'topic_progress'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user_profiles.id'))
    topic = Column(String(100), nullable=False)
    easy = Column(Integer, default=0)
    medium = Column(Integer, default=0)
    hard = Column(Integer, default=0)
    user_profile = relationship("UserProfile", back_populates="progress")

# Weakness Model
class Weakness(Base):
    __tablename__ = 'weaknesses'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user_profiles.id'))
    weakness = Column(Text, nullable=False)
    user_profile = relationship("UserProfile", back_populates="weaknesses")

# Create tables if not already created
Base.metadata.create_all(engine)

# Database session management
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Functions for interacting with DB
def create_or_update_user_profile(db, user_id: str, name: str):
    user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not user:
        user = UserProfile(user_id=user_id, name=name)
        db.add(user)
        db.commit()
    return user

def update_topic_progress(db, user_id: str, topic: str, difficulty: str):
    user = create_or_update_user_profile(db, user_id, name="")  # Pass name as empty for simplicity
    topic_progress = db.query(TopicProgress).filter(TopicProgress.user_id == user.id, TopicProgress.topic == topic).first()
    if not topic_progress:
        topic_progress = TopicProgress(user_id=user.id, topic=topic)
        db.add(topic_progress)

    if difficulty == 'easy':
        topic_progress.easy += 1
    elif difficulty == 'medium':
        topic_progress.medium += 1
    elif difficulty == 'hard':
        topic_progress.hard += 1
    db.commit()

def add_weakness(db, user_id: str, weakness: str):
    user = create_or_update_user_profile(db, user_id, name="")  # Pass name as empty for simplicity
    weakness_entry = Weakness(user_id=user.id, weakness=weakness)
    db.add(weakness_entry)
    db.commit()
