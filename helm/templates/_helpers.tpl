{{- define "studio-chatbot.name" -}}
studio-chatbot
{{- end -}}

{{- define "studio-chatbot.selectorLabels" -}}
app.kubernetes.io/name: {{ include "studio-chatbot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "studio-chatbot.labels" -}}
{{ include "studio-chatbot.selectorLabels" . }}
environment: {{ .Values.environment }}
{{- end -}}
