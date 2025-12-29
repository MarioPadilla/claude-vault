---
date: '2025-12-28T10:30:05+00:00'
message_count: 6
tags:
- code-session
- python
- javascript
- debugging
- api
title: '[web-api] Help me create a basic Flask API with error handling'
updated: '2025-12-28T10:31:25+00:00'
uuid: example_session_abc123
---

## **user** *(2025-12-28 10:30)*

Help me create a basic Flask API with error handling

---

## 🤖 Claude *(2025-12-28 10:30)*

I'll help you create a Flask API with proper error handling. Let me start with a basic structure.

---

## 🤖 Claude *(2025-12-28 10:30)*

**[Tool: WriteFile]**
```json
{
  "path": "app.py",
  "content": "from flask import Flask, jsonify, request\nfrom functools import wraps\n\napp = Flask(__name__)\n\n# Custom error handlers\n@app.errorhandler(404)\ndef not_found(error):\n    return jsonify({'error': 'Resource not found'}), 404\n\n@app.errorhandler(500)\ndef internal_error(error):\n    return jsonify({'error': 'Internal server error'}), 500\n\n# Validation decorator\ndef validate_json(*expected_fields):\n    def decorator(f):\n        @wraps(f)\n        def wrapper(*args, **kwargs):\n            if not request.json:\n                return jsonify({'error': 'No JSON data provided'}), 400\n            missing = [field for field in expected_fields if field not in request.json]\n            if missing:\n                return jsonify({'error': f'Missing fields: {missing}'}), 400\n            return f(*args, **kwargs)\n        return wrapper\n    return decorator\n\n# Example endpoint with validation\n@app.route('/api/users', methods=['POST'])\n@validate_json('name', 'email')\ndef create_user():\n    data = request.json\n    return jsonify({'message': 'User created', 'user': data}), 201\n\nif __name__ == '__main__':\n    app.run(debug=True)"
}
```

---

## **user** *(2025-12-28 10:31)*

Can you also add logging for debugging purposes?

---

## 🤖 Claude *(2025-12-28 10:31)*

Great idea! Let me add comprehensive logging to track requests and errors.

---

## 🤖 Claude *(2025-12-28 10:31)*

**[Tool: WriteFile]**
```json
{
  "path": "logger.py",
  "content": "import logging\nfrom logging.handlers import RotatingFileHandler\nimport os\n\ndef setup_logger(app):\n    \"\"\"Configure application logging\"\"\"\n    \n    # Create logs directory if it doesn't exist\n    if not os.path.exists('logs'):\n        os.mkdir('logs')\n    \n    # Set up file handler with rotation\n    file_handler = RotatingFileHandler(\n        'logs/app.log',\n        maxBytes=10240000,\n        backupCount=10\n    )\n    file_handler.setFormatter(logging.Formatter(\n        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'\n    ))\n    file_handler.setLevel(logging.INFO)\n    \n    # Add handler to app logger\n    app.logger.addHandler(file_handler)\n    app.logger.setLevel(logging.INFO)\n    app.logger.info('Application startup')\n    \n    return app"
}
```