import axios from "axios";
import { API_BASE as API } from "./client";

export const askRag = (patient_id, doctor_id, query) =>
  axios.post(`${API}/ask`, { patient_id, doctor_id, query });

export const ocrImage = (file) => {
  const form = new FormData();
  form.append("file", file);
  return axios.post(`${API}/ocr/image`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const sttAudio = (file) => {
  const form = new FormData();
  form.append("file", file);
  return axios.post(`${API}/stt/audio`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};
