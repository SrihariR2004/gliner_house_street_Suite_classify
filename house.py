from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from gliner import GLiNER  # Assuming GLiNER is imported from gliner

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input_text: InputText):
    text = input_text.text

    label = ["house_no", "street_no", "Suite_no"]
    model = GLiNER.from_pretrained("srihari420/house_street_name_classifier")
    prediction = model.predict_entity(text, label=label)
    out = [{"text": i['text'], "label": i['label']} for i in prediction]
    return {"prediction": out}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)