#!/usr/bin/env bash
set -euo pipefail

space_arn="${1:?usage: $0 SPACE_ARN [AWS_PROFILE]}"
aws_profile="${2:-}"

if [[ "$space_arn" =~ ^arn:aws[a-z-]*:sagemaker:([a-z0-9-]+):[0-9]{12}:space/[^/]+/[^/]+$ ]]; then
    region="${BASH_REMATCH[1]}"
else
    printf 'invalid SageMaker Space ARN: %s\n' "$space_arn" >&2
    exit 2
fi

command -v aws >/dev/null
command -v jq >/dev/null
command -v session-manager-plugin >/dev/null

aws_args=(
    sagemaker start-session
    --resource-identifier "$space_arn"
    --region "$region"
)
if [[ -n "$aws_profile" ]]; then
    aws_args+=(--profile "$aws_profile")
fi

response="$(aws "${aws_args[@]}")"
session="$(
    jq -ce '
        {
            streamUrl: .StreamUrl,
            tokenValue: .TokenValue,
            sessionId: .SessionId
        }
        | select(all(.[]; type == "string" and length > 0))
    ' <<<"$response"
)"

exec session-manager-plugin "$session" "$region" StartSession
