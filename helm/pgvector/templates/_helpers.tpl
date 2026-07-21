{{- define "pgvector.name" -}}
pgvector
{{- end -}}

{{- define "pgvector.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pgvector.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "pgvector.labels" -}}
{{ include "pgvector.selectorLabels" . }}
environment: {{ .Values.environment }}
{{- end -}}
