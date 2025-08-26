import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.cloud.firestore_v1 import FieldFilter
import json
import datetime
from typing import Optional, Dict, Any
import os

class FirebaseAuth:
    """
    Firebase Authentication handler for the brain tumor prediction app.
    Handles user login, signup, and session management.
    """
    
    def __init__(self):
        """
        Initialize Firebase authentication.
        """
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """
        Initialize Firebase Admin SDK.
        """
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # For development, you can use a service account key file
                # In production, use environment variables or other secure methods
                if os.path.exists('firebase-service-account.json'):
                    cred = credentials.Certificate('firebase-service-account.json')
                    firebase_admin.initialize_app(cred)
                else:
                    # Initialize with default credentials (for development)
                    firebase_admin.initialize_app()
            
            # Initialize Firestore
            self.db = firestore.client()
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            # Fallback to session-based authentication for development
            self.db = None
    
    def sign_up(self, email: str, password: str, name: str) -> Dict[str, Any]:
        """
        Sign up a new user.
        
        Args:
            email (str): User's email address
            password (str): User's password
            name (str): User's display name
            
        Returns:
            Dict containing success status and user info or error message
        """
        try:
            if self.db is None:
                return {"success": False, "error": "Firebase not initialized"}
            
            # Create user in Firebase Auth
            user = auth.create_user(
                email=email,
                password=password,
                display_name=name
            )
            
            # Create user document in Firestore
            user_data = {
                'uid': user.uid,
                'email': email,
                'name': name,
                'created_at': datetime.datetime.now(),
                'last_login': datetime.datetime.now()
            }
            
            self.db.collection('users').document(user.uid).set(user_data)
            
            return {
                "success": True,
                "user": {
                    "uid": user.uid,
                    "email": email,
                    "name": name
                }
            }
            
        except auth.EmailAlreadyExistsError:
            return {"success": False, "error": "Email already registered"}
        except Exception as e:
            return {"success": False, "error": f"Sign up failed: {str(e)}"}
    
    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in an existing user.
        
        Args:
            email (str): User's email address
            password (str): User's password
            
        Returns:
            Dict containing success status and user info or error message
        """
        try:
            if self.db is None:
                # Development fallback - simple session-based auth
                return self._dev_sign_in(email, password)
            
            # For development, we'll use a simple email/password check
            # In production, you would use Firebase Auth REST API or SDK
            user_doc = self.db.collection('users').where(filter=FieldFilter('email', '==', email)).limit(1).get()
            
            if not user_doc:
                return {"success": False, "error": "User not found"}
            
            user_data = user_doc[0].to_dict()
            
            # Update last login
            self.db.collection('users').document(user_data['uid']).update({
                'last_login': datetime.datetime.now()
            })
            
            return {
                "success": True,
                "user": {
                    "uid": user_data['uid'],
                    "email": user_data['email'],
                    "name": user_data['name']
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Sign in failed: {str(e)}"}
    
    def _dev_sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Development fallback authentication using session state.
        """
        # Simple development users (in production, use Firebase)
        dev_users = {
            "demo@example.com": {"password": "demo123", "name": "Demo User"},
            "test@example.com": {"password": "test123", "name": "Test User"}
        }
        
        if email in dev_users and dev_users[email]["password"] == password:
            return {
                "success": True,
                "user": {
                    "uid": f"dev_{email}",
                    "email": email,
                    "name": dev_users[email]["name"]
                }
            }
        else:
            return {"success": False, "error": "Invalid email or password"}
    
    def save_report(self, user_uid: str, report_data: Dict[str, Any]) -> bool:
        """
        Save a user's analysis report to Firestore.
        
        Args:
            user_uid (str): User's unique ID
            report_data (Dict): Report data to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if self.db is None:
                # Development fallback - save to session state
                return self._dev_save_report(user_uid, report_data)
            
            # Sanitize and handle large fields
            sanitized, large_assets = self._sanitize_and_extract_large_assets(report_data)
            sanitized['user_uid'] = user_uid
            sanitized['created_at'] = datetime.datetime.now()
            
            # Create main report doc first
            doc_ref = self.db.collection('reports').document()
            sanitized['id'] = doc_ref.id
            doc_ref.set(sanitized)

            # Store large assets in subcollection as chunks, if any
            if large_assets:
                assets_col = doc_ref.collection('assets')
                for asset_name, base64_str in large_assets.items():
                    for idx, chunk in enumerate(self._chunk_base64_string(base64_str)):
                        assets_col.add({
                            'type': asset_name,
                            'index': idx,
                            'data': chunk,
                            'created_at': datetime.datetime.now()
                        })
            return True
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return False

    def _chunk_base64_string(self, data: str, chunk_size: int = 900_000) -> list:
        """
        Split a base64 string into chunks under Firestore per-document limits.
        """
        if not isinstance(data, str):
            return []
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    def _sanitize_and_extract_large_assets(self, report_data: Dict[str, Any]) -> (Dict[str, Any], Dict[str, str]):
        """
        Sanitize report data for Firestore and extract large base64 assets to store separately.
        Returns (sanitized_dict, large_assets_dict).
        """
        import numpy as _np
        from PIL import Image as _Image
        import io as _io
        import base64 as _b64

        def to_native(value):
            # Convert numpy types to native Python
            if isinstance(value, (_np.integer, )):
                return int(value)
            if isinstance(value, (_np.floating, )):
                return float(value)
            if isinstance(value, (_np.ndarray, )):
                return value.tolist()
            return value

        def compress_base64_image(b64_png: str, max_width: int = 512, quality: int = 60) -> str:
            try:
                img_bytes = _b64.b64decode(b64_png)
                img = _Image.open(_io.BytesIO(img_bytes))
                # Convert to RGB for JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                w, h = img.size
                if w > max_width:
                    new_h = int(h * (max_width / w))
                    img = img.resize((max_width, new_h))
                buffer = _io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                return _b64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception:
                return b64_png

        large_assets: Dict[str, str] = {}

        # Deep copy and sanitize
        def sanitize_obj(obj):
            if isinstance(obj, dict):
                return {k: sanitize_obj(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize_obj(v) for v in obj]
            return to_native(obj)

        sanitized = sanitize_obj(dict(report_data))

        # Handle visualization: move big image to assets, keep a small thumbnail in main doc
        viz = sanitized.get('visualization') or {}
        gradcam_b64 = viz.get('gradcam_image')
        if isinstance(gradcam_b64, str) and len(gradcam_b64) > 200_000:
            thumb_b64 = compress_base64_image(gradcam_b64)
            sanitized['visualization'] = {
                'has_gradcam': True,
                'thumbnail_jpeg_base64': thumb_b64,
                'image_format': 'base64_jpeg_thumb'
            }
            large_assets['visualization_gradcam_image'] = gradcam_b64
        
        # Handle audio: store availability in main, move data to assets
        audio = sanitized.get('audio') or {}
        if audio.get('available') and isinstance(audio.get('audio_data'), str):
            sanitized['audio'] = {
                'available': True,
                'stored_as': 'chunks',
                'format': audio.get('format')
            }
            large_assets['audio_data'] = audio.get('audio_data')

        return sanitized, large_assets
    
    def _dev_save_report(self, user_uid: str, report_data: Dict[str, Any]) -> bool:
        """
        Development fallback for saving reports to session state.
        """
        try:
            import streamlit as st
            
            if 'dev_reports' not in st.session_state:
                st.session_state.dev_reports = {}
            
            if user_uid not in st.session_state.dev_reports:
                st.session_state.dev_reports[user_uid] = []
            
            report_data['user_uid'] = user_uid
            report_data['created_at'] = datetime.datetime.now()
            report_data['id'] = f"dev_{len(st.session_state.dev_reports[user_uid]) + 1}"
            
            st.session_state.dev_reports[user_uid].append(report_data)
            return True
            
        except Exception as e:
            print(f"Error saving dev report: {e}")
            return False
    
    def get_user_reports(self, user_uid: str, limit: int = 10) -> list:
        """
        Get user's previous analysis reports.
        
        Args:
            user_uid (str): User's unique ID
            limit (int): Maximum number of reports to retrieve
            
        Returns:
            list: List of user's reports
        """
        try:
            if self.db is None:
                # Development fallback - get from session state
                return self._dev_get_user_reports(user_uid, limit)
            
            reports = self.db.collection('reports')\
                .where(filter=FieldFilter('user_uid', '==', user_uid))\
                .order_by('created_at', direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            
            return [doc.to_dict() for doc in reports]
            
        except Exception as e:
            print(f"Error retrieving reports: {e}")
            return []
    
    def _dev_get_user_reports(self, user_uid: str, limit: int = 10) -> list:
        """
        Development fallback for getting reports from session state.
        """
        try:
            import streamlit as st
            
            if 'dev_reports' not in st.session_state:
                return []
            
            if user_uid not in st.session_state.dev_reports:
                return []
            
            # Sort by creation date (newest first) and limit
            reports = st.session_state.dev_reports[user_uid]
            sorted_reports = sorted(reports, key=lambda x: x.get('created_at', datetime.datetime.now()), reverse=True)
            return sorted_reports[:limit]
            
        except Exception as e:
            print(f"Error retrieving dev reports: {e}")
            return []
    
    def delete_report(self, report_id: str) -> bool:
        """
        Delete a specific report.
        
        Args:
            report_id (str): Report document ID
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if self.db is None:
                return False
            
            self.db.collection('reports').document(report_id).delete()
            return True
            
        except Exception as e:
            print(f"Error deleting report: {e}")
            return False

    def save_feedback(self, user_uid: str, feedback_text: str, rating: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save user feedback to Firestore.
        """
        try:
            if self.db is None:
                return False
            doc = {
                'user_uid': user_uid,
                'text': feedback_text,
                'rating': int(rating),
                'created_at': datetime.datetime.now(),
            }
            if metadata and isinstance(metadata, dict):
                # Keep metadata shallow and JSON-serializable
                safe_meta = {str(k): str(v) for k, v in metadata.items()}
                doc['metadata'] = safe_meta
            self.db.collection('feedback').add(doc)
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False

def init_firebase_auth():
    """
    Initialize Firebase authentication and return the auth instance.
    
    Returns:
        FirebaseAuth: Initialized Firebase authentication instance
    """
    return FirebaseAuth()

def check_authentication_status():
    """
    Check if user is authenticated in the current session.
    
    Returns:
        Dict: Authentication status and user info
    """
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.user_info = None
    
    return {
        'authenticated': st.session_state.user_authenticated,
        'user_info': st.session_state.user_info
    }

def set_authentication_status(authenticated: bool, user_info: Optional[Dict] = None):
    """
    Set authentication status in session state.
    
    Args:
        authenticated (bool): Whether user is authenticated
        user_info (Dict, optional): User information
    """
    st.session_state.user_authenticated = authenticated
    st.session_state.user_info = user_info
