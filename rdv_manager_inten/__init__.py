import azure.functions as func
from rank_bm25 import BM25Okapi
import numpy as np
import re
from unidecode import unidecode
import logging
import json
import nltk
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BM25Analyzer:
    def __init__(self):
        self.stemmer = SnowballStemmer("french")
        self.stopwords = list(stopwords.words('french'))
        
        # Exemples structurés avec variantes
        self.corpus = {
            "prendre": [
                      "Je veux fixer un rendez-vous", "Je souhaite réserver une consultation",
                      "Est-il possible de programmer un examen ?", "Planifier un créneau pour une imagerie",
                      "J’aimerais obtenir une date pour un rendez-vous médical", "Je voudrais caler un rendez-vous en urgence",
                      "Peut-on organiser une consultation pour moi ?", "Comment prendre un rendez-vous rapidement ?",
                      "Réserver une place pour un contrôle médical", "Je cherche à planifier un examen d’imagerie",
                      "Je dois prendre un créneau pour une IRM", "Je veux m’inscrire pour un scanner",
                      "Pourrais-je bloquer une date chez le radiologue ?", "J’ai besoin d’un créneau libre au plus vite",
                      "Je voudrais m’enregistrer pour une consultation", "Comment fixer un rendez-vous en ligne ?",
                      "Est-il possible de choisir un horaire pour un examen ?", "Je cherche un créneau disponible pour une échographie",
                      "Je veux demander un rendez-vous médical", "Trouver un créneau pour une consultation",
                      "Je dois absolument voir un spécialiste en imagerie", "Est-ce que je peux réserver une visite médicale ?","Je veux réserver un examen de radiologie."
                  ],
                  "modifier": [
                      "Je veux ajuster la date de mon rendez-vous", "Reporter mon examen médical",
                      "Comment changer l’horaire de ma consultation ?", "Je dois repousser mon rendez-vous à une autre date",
                      "Puis-je revoir l’organisation de mon rendez-vous ?", "Décaler mon passage chez le radiologue","Je veux décaler l’heure de mon rendez-vous",
                      "Je souhaite reprogrammer mon examen de radiologie", "Je veux modifier l’heure de mon rendez-vous",
                      "Est-il possible de déplacer mon scanner à une autre journée ?", "Modifier la plage horaire de mon IRM",
                      "Je veux avancer mon rendez-vous si possible", "Je dois faire correspondre mon rendez-vous à mon emploi du temps",
                      "Puis-je ajuster la réservation de mon examen ?", "Changer le créneau de ma consultation médicale",
                      "Je veux trouver un moment plus adapté pour mon rendez-vous", "Est-il possible de sélectionner une nouvelle date ?",
                      "Je préfère un autre jour pour mon imagerie médicale", "J’aimerais modifier les détails de ma réservation",
                      "Comment faire pour actualiser mon rendez-vous ?", "Déplacer ma consultation à une date ultérieure"
                  ],
                  "annuler": [
                      "Je veux supprimer mon rendez-vous", "Annuler ma prise de rendez-vous",
                      "Comment résilier ma consultation ?", "Je dois mettre fin à ma réservation",
                      "Je préfère ne plus maintenir mon rendez-vous médical", "Je veux stopper mon rendez-vous prévu",
                      "Peut-on effacer mon examen de radiologie ?", "Je souhaite désactiver ma réservation",
                      "Puis-je interrompre ma consultation planifiée ?", "J’aimerais me désengager de mon rendez-vous",
                      "Je ne pourrai pas être présent, comment annuler ?", "Annulation de mon passage chez le spécialiste",
                      "Je dois annuler mon scanner prévu", "Je veux clôturer ma réservation d’imagerie",
                      "Je souhaite abandonner mon rendez-vous médical", "Retirer mon nom de la liste des rendez-vous",
                      "Supprimer mon créneau chez le radiologue", "Je n’ai plus besoin de cette consultation, comment annuler ?",
                      "Je veux annuler sans frais, est-ce possible ?", "Faire disparaître mon rendez-vous du planning"
                  ],
                  "consulter": [
                      "Je veux accéder à mes rendez-vous", "Quels sont les prochains examens programmés ?",
                      "Afficher mon planning médical", "Lister mes consultations prévues",
                      "Où puis-je voir mes rendez-vous à venir ?", "Vérifier l’état de mes réservations médicales",
                      "Accéder aux détails de mon prochain rendez-vous", "Je veux examiner mon historique de consultations",
                      "Comment visualiser mes rendez-vous confirmés ?", "Je souhaite contrôler mon agenda médical",
                      "Peut-on me rappeler la date de mon examen ?", "Je voudrais récupérer mes informations de rendez-vous",
                      "Afficher mes créneaux en imagerie", "Voir les horaires de mes consultations futures",
                      "Je veux inspecter la liste de mes examens prévus", "Peut-on me donner un aperçu de mes rendez-vous ?",
                      "Je veux obtenir un récapitulatif de mon agenda médical", "Retrouver les dates de mes prochains passages",
                      "Consulter l’ensemble de mes examens planifiés", "Comment voir mes prises de rendez-vous antérieures et futures ?"
                  ]
        }
        
        self._preprocess_corpus()
        self.bm25 = self._init_bm25()
    
    def _preprocess(self, text):
        """Nettoyage profond du texte avec stemming"""
        try:
            text = unidecode(text.lower())
            text = re.sub(r'[^\w\s]', '', text)
            tokens = [self.stemmer.stem(word) for word in text.split() 
                     if word not in self.stopwords and len(word) > 2]
            return tokens
        except Exception as e:
            logger.error(f"Erreur prétraitement : {e}")
            return []
    
    def _preprocess_corpus(self):
        """Préparation du corpus d'entraînement"""
        self.processed_corpus = []
        self.category_mapping = []
        
        for category, docs in self.corpus.items():
            for doc in docs:
                processed = self._preprocess(doc)
                self.processed_corpus.append(processed)
                self.category_mapping.append(category)
    
    def _init_bm25(self):
        """Initialisation du modèle BM25 avec paramètres optimisés"""
        return BM25Okapi(
            self.processed_corpus,
            k1=1.6,  # Contrôle la saturation de fréquence des termes
            b=0.75   # Contrôle l'impact de la longueur du document
        )
    
    def get_intent(self, query, confidence_threshold=0.2):
        """Détection d'intention avec seuil de confiance"""
        try:
            processed_query = self._preprocess(query)
            scores = self.bm25.get_scores(processed_query)
            
            best_score_idx = np.argmax(scores)
            best_score = scores[best_score_idx]
            
            if best_score < confidence_threshold:
                return "Intention inconnue"
            
            return self.category_mapping[best_score_idx]
        
        except Exception as e:
            logger.error(f"Erreur détection : {e}")
            return "Intention inconnue"


# Utilisation
analyzer = BM25Analyzer()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        query = req_body.get('text')

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "No query provided in request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        result = analyzer.get_intent(query)

        return func.HttpResponse(
            json.dumps({"response": result}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
