from AIGamePlatform.CompetitionSocket import CompetitionSocket
from google_auth_oauthlib.flow import InstalledAppFlow

class Othello():
    def __init__(self):            
        flow = InstalledAppFlow.from_client_secrets_file(
            'AIGamePlatform/client_secret.json',
            scopes=['openid', 'https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email'])
        flow.run_local_server(port=9486)
        self.token = flow.credentials._id_token
        
    def competition(self, competition_id):
        def decorator(f):
            CompetitionSocket(competition_id, self.token, f)
        return decorator


