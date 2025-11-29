from app import app, db, DetectionHistory
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError

# Run this script to add the trusted_source column to the detection_history table

with app.app_context():
    try:
        # Check if the column already exists
        inspector = sa.inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('detection_history')]
        
        if 'trusted_source' not in columns:
            print("Adding trusted_source column to detection_history table...")
            # Add the column using the correct SQLAlchemy syntax for SQLite
            with db.engine.connect() as conn:
                conn.execute(sa.text('ALTER TABLE detection_history ADD COLUMN trusted_source BOOLEAN DEFAULT 0'))
                conn.commit()
            print("Column added successfully!")
        else:
            print("Column trusted_source already exists.")
            
    except OperationalError as e:
        print(f"Error: {e}")
        print("\nTrying alternative approach...")
        try:
            # SQLite has limited ALTER TABLE support, so we might need to recreate the table
            # First, create a backup of the current data
            with db.engine.connect() as conn:
                conn.execute(sa.text('CREATE TABLE detection_history_backup AS SELECT * FROM detection_history'))
                
                # Drop the original table
                conn.execute(sa.text('DROP TABLE detection_history'))
                conn.commit()
            
            # Recreate the table with the new column
            db.create_all()
            
            # Copy the data back
            with db.engine.connect() as conn:
                conn.execute(sa.text('INSERT INTO detection_history (id, user_id, subject, prediction, timestamp, detected_language, translated_text) SELECT id, user_id, subject, prediction, timestamp, detected_language, translated_text FROM detection_history_backup'))
                
                # Drop the backup table
                conn.execute(sa.text('DROP TABLE detection_history_backup'))
                conn.commit()
            
            print("Table recreated with the new column successfully!")
        except Exception as e2:
            print(f"Error in alternative approach: {e2}")
            print("Please backup your data and recreate the database.")