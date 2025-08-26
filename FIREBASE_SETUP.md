# Firebase Setup Guide

This guide will help you set up Firebase authentication and Firestore database for the Brain Tumor Detection AI application.

## Prerequisites

1. A Google account
2. Python 3.8+ installed
3. The brain tumor prediction application files

## Step 1: Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or "Add project"
3. Enter a project name (e.g., "brain-tumor-detection-ai")
4. Choose whether to enable Google Analytics (optional)
5. Click "Create project"

## Step 2: Enable Authentication

1. In your Firebase project, go to "Authentication" in the left sidebar
2. Click "Get started"
3. Go to the "Sign-in method" tab
4. Enable "Email/Password" authentication:
   - Click on "Email/Password"
   - Toggle "Enable"
   - Click "Save"

## Step 3: Set Up Firestore Database

1. In your Firebase project, go to "Firestore Database" in the left sidebar
2. Click "Create database"
3. Choose "Start in test mode" (for development)
4. Select a location for your database (choose the closest to your users)
5. Click "Done"

## Step 4: Create Service Account

1. In your Firebase project, go to "Project settings" (gear icon)
2. Go to the "Service accounts" tab
3. Click "Generate new private key"
4. Download the JSON file
5. Rename it to `firebase-service-account.json`
6. Place it in your project root directory

## Step 5: Configure Security Rules (Optional)

For production, you should configure Firestore security rules. In the Firestore Database:

1. Go to the "Rules" tab
2. Replace the default rules with:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Users can only access their own reports
    match /reports/{reportId} {
      allow read, write: if request.auth != null && 
        request.auth.uid == resource.data.user_uid;
    }
  }
}
```

## Step 6: Install Dependencies

Install the required Python packages:

```bash
pip install firebase-admin streamlit-authenticator
```

## Step 7: Test the Setup

1. Start your Flask backend:
   ```bash
   python backend.py
   ```

2. Start your Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

3. Open the application in your browser
4. Try creating a new account and logging in

## Troubleshooting

### Firebase Not Initialized Error

If you see "Firebase not initialized" errors:

1. Check that `firebase-service-account.json` is in the project root
2. Verify the JSON file is valid and contains all required fields
3. Ensure the service account has the necessary permissions

### Authentication Issues

If users can't sign up or log in:

1. Check that Email/Password authentication is enabled in Firebase
2. Verify the Firestore database is created and accessible
3. Check the Firebase console for any error messages

### Database Connection Issues

If reports aren't being saved:

1. Verify Firestore is enabled in your Firebase project
2. Check that the service account has Firestore permissions
3. Review the security rules if you've customized them

## Development vs Production

### Development Mode

For development, the app will work with session-based authentication if Firebase is not configured. This allows you to test the application without setting up Firebase.

### Production Mode

For production deployment:

1. Use a proper Firebase service account
2. Configure appropriate security rules
3. Enable additional authentication methods if needed
4. Set up proper error monitoring

## Environment Variables (Alternative)

Instead of using a service account file, you can use environment variables:

```bash
export FIREBASE_PROJECT_ID="your-project-id"
export FIREBASE_PRIVATE_KEY="your-private-key"
export FIREBASE_CLIENT_EMAIL="your-client-email"
```

Then modify `firebase_auth.py` to use these environment variables.

## Security Best Practices

1. **Never commit** `firebase-service-account.json` to version control
2. Use environment variables in production
3. Configure proper Firestore security rules
4. Regularly rotate service account keys
5. Monitor Firebase usage and costs

## Support

If you encounter issues:

1. Check the Firebase console for error messages
2. Review the application logs
3. Verify all dependencies are installed
4. Ensure your Firebase project is properly configured

For additional help, refer to the [Firebase documentation](https://firebase.google.com/docs).
