import os
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

IS_DEBUG = False

# Save information on AWS using secert
S3_BUCKET_NAME = 'cat-recognizer-images'
S3_BUCKET_BASE_URL = 'https://s3-ap-northeast-1.amazonaws.com/cat-recognizer-images/'
AWS_ACCESS_KEY_ID = 'AKIAINVSEP6T2PPIXG5A'
AWS_SECRET_ACCESS_KEY = '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'
