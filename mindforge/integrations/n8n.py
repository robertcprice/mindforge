"""
MindForge n8n Integration

Control n8n workflows programmatically - the consciousness engine's "body"
for automating real-world tasks.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import httpx

from mindforge.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecution:
    """Result of a workflow execution."""

    execution_id: str
    workflow_id: str
    workflow_name: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    data: Optional[dict] = None
    error: Optional[str] = None


class N8NClient:
    """Client for interacting with n8n API.

    n8n serves as MindForge's "body" - it can trigger automations,
    send notifications, interact with external services, etc.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5678",
        api_key: Optional[str] = None,
        container_name: str = "n8n",
    ):
        """Initialize n8n client.

        Args:
            base_url: n8n instance URL
            api_key: API key (from N8N_API_KEY env var if not provided)
            container_name: Docker container name for n8n
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1"
        self.api_key = api_key or os.getenv("N8N_API_KEY")
        self.container_name = container_name

        self._client = httpx.Client(
            base_url=self.api_url,
            headers=self._get_headers(),
            timeout=30.0,
        )

    def _get_headers(self) -> dict:
        """Get HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-N8N-API-KEY"] = self.api_key
        return headers

    def is_healthy(self) -> bool:
        """Check if n8n is running and accessible."""
        try:
            response = self._client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"n8n health check failed: {e}")
            return False

    def is_container_running(self) -> bool:
        """Check if n8n Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip().lower() == "true"
        except Exception as e:
            logger.warning(f"Docker check failed: {e}")
            return False

    def start_container(self) -> bool:
        """Start the n8n Docker container."""
        try:
            result = subprocess.run(
                ["docker", "start", self.container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to start n8n container: {e}")
            return False

    def stop_container(self) -> bool:
        """Stop the n8n Docker container."""
        try:
            result = subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to stop n8n container: {e}")
            return False

    def list_workflows(self, active_only: bool = False) -> list[dict]:
        """List all workflows.

        Args:
            active_only: Only return active workflows

        Returns:
            List of workflow metadata
        """
        try:
            response = self._client.get("/workflows")
            response.raise_for_status()
            workflows = response.json().get("data", [])

            if active_only:
                workflows = [w for w in workflows if w.get("active")]

            return workflows
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return []

    def get_workflow(self, workflow_id: str) -> Optional[dict]:
        """Get a specific workflow by ID."""
        try:
            response = self._client.get(f"/workflows/{workflow_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None

    def execute_workflow(
        self,
        workflow_id: str,
        data: Optional[dict] = None,
        wait: bool = True,
    ) -> WorkflowExecution:
        """Execute a workflow.

        Args:
            workflow_id: ID of workflow to execute
            data: Input data for the workflow
            wait: Wait for execution to complete

        Returns:
            WorkflowExecution result
        """
        try:
            payload = {}
            if data:
                payload["data"] = data

            response = self._client.post(
                f"/workflows/{workflow_id}/execute",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            return WorkflowExecution(
                execution_id=result.get("executionId", ""),
                workflow_id=workflow_id,
                workflow_name=result.get("workflowData", {}).get("name", "Unknown"),
                status=result.get("status", "unknown"),
                started_at=datetime.now(),
                data=result.get("data"),
            )
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            return WorkflowExecution(
                execution_id="",
                workflow_id=workflow_id,
                workflow_name="Unknown",
                status="error",
                started_at=datetime.now(),
                error=str(e),
            )

    def trigger_webhook(
        self,
        webhook_path: str,
        data: Optional[dict] = None,
        method: str = "POST",
    ) -> dict:
        """Trigger a webhook-based workflow.

        Args:
            webhook_path: Path portion of webhook URL
            data: Payload to send
            method: HTTP method (POST, GET)

        Returns:
            Webhook response
        """
        try:
            webhook_url = f"{self.base_url}/webhook/{webhook_path}"

            if method.upper() == "GET":
                response = httpx.get(webhook_url, params=data, timeout=30)
            else:
                response = httpx.post(webhook_url, json=data, timeout=30)

            response.raise_for_status()
            return {"success": True, "response": response.json()}
        except Exception as e:
            logger.error(f"Webhook trigger failed: {e}")
            return {"success": False, "error": str(e)}

    def activate_workflow(self, workflow_id: str) -> bool:
        """Activate a workflow."""
        try:
            response = self._client.patch(
                f"/workflows/{workflow_id}",
                json={"active": True},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to activate workflow {workflow_id}: {e}")
            return False

    def deactivate_workflow(self, workflow_id: str) -> bool:
        """Deactivate a workflow."""
        try:
            response = self._client.patch(
                f"/workflows/{workflow_id}",
                json={"active": False},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate workflow {workflow_id}: {e}")
            return False

    def get_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get workflow execution history.

        Args:
            workflow_id: Filter by workflow
            status: Filter by status (success, error, waiting)
            limit: Maximum results

        Returns:
            List of executions
        """
        try:
            params = {"limit": limit}
            if workflow_id:
                params["workflowId"] = workflow_id
            if status:
                params["status"] = status

            response = self._client.get("/executions", params=params)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logger.error(f"Failed to get executions: {e}")
            return []

    def close(self):
        """Close the HTTP client."""
        self._client.close()


class N8NTool(Tool):
    """Tool for controlling n8n workflows.

    This is how MindForge interacts with the external world through
    automated workflows.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5678",
        api_key: Optional[str] = None,
        container_name: str = "n8n",
    ):
        """Initialize n8n tool."""
        super().__init__(
            name="n8n",
            description="Control n8n workflow automation - list, execute, and manage workflows",
            requires_confirmation=False,  # Most operations are safe
        )

        self.client = N8NClient(
            base_url=base_url,
            api_key=api_key,
            container_name=container_name,
        )

    def execute(
        self,
        operation: str,
        workflow_id: Optional[str] = None,
        webhook_path: Optional[str] = None,
        data: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute an n8n operation.

        Operations:
            - health: Check if n8n is running
            - list: List workflows
            - get: Get workflow details
            - run: Execute a workflow
            - webhook: Trigger a webhook
            - activate: Activate a workflow
            - deactivate: Deactivate a workflow
            - history: Get execution history
            - start: Start n8n container
            - stop: Stop n8n container

        Args:
            operation: Operation to perform
            workflow_id: Workflow ID (for get, run, activate, deactivate)
            webhook_path: Webhook path (for webhook operation)
            data: Data payload (for run, webhook)
            **kwargs: Additional arguments
        """
        import time
        start_time = time.time()

        try:
            if operation == "health":
                healthy = self.client.is_healthy()
                container_running = self.client.is_container_running()
                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"n8n API: {'healthy' if healthy else 'unreachable'}, Container: {'running' if container_running else 'stopped'}",
                    metadata={"api_healthy": healthy, "container_running": container_running},
                )

            elif operation == "list":
                active_only = kwargs.get("active_only", False)
                workflows = self.client.list_workflows(active_only=active_only)

                output_lines = [f"Found {len(workflows)} workflow(s):"]
                for wf in workflows:
                    status = "✓" if wf.get("active") else "○"
                    output_lines.append(f"  {status} [{wf.get('id')}] {wf.get('name')}")

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="\n".join(output_lines),
                    metadata={"count": len(workflows), "workflows": workflows},
                )

            elif operation == "get":
                if not workflow_id:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="workflow_id required for 'get' operation",
                    )

                workflow = self.client.get_workflow(workflow_id)
                if workflow:
                    result = ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Workflow: {workflow.get('name')}\nActive: {workflow.get('active')}\nNodes: {len(workflow.get('nodes', []))}",
                        metadata=workflow,
                    )
                else:
                    result = ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=f"Workflow {workflow_id} not found",
                    )

            elif operation == "run":
                if not workflow_id:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="workflow_id required for 'run' operation",
                    )

                execution = self.client.execute_workflow(workflow_id, data=data)

                if execution.error:
                    result = ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=execution.error,
                    )
                else:
                    result = ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Executed workflow {execution.workflow_name} (ID: {execution.execution_id})",
                        metadata={
                            "execution_id": execution.execution_id,
                            "status": execution.status,
                            "data": execution.data,
                        },
                    )

            elif operation == "webhook":
                if not webhook_path:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="webhook_path required for 'webhook' operation",
                    )

                method = kwargs.get("method", "POST")
                response = self.client.trigger_webhook(webhook_path, data=data, method=method)

                if response.get("success"):
                    result = ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Webhook triggered successfully",
                        metadata=response,
                    )
                else:
                    result = ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=response.get("error", "Unknown webhook error"),
                    )

            elif operation == "activate":
                if not workflow_id:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="workflow_id required for 'activate' operation",
                    )

                success = self.client.activate_workflow(workflow_id)
                result = ToolResult(
                    status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                    output=f"Workflow {workflow_id} activated" if success else "",
                    error=None if success else f"Failed to activate workflow {workflow_id}",
                )

            elif operation == "deactivate":
                if not workflow_id:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error="workflow_id required for 'deactivate' operation",
                    )

                success = self.client.deactivate_workflow(workflow_id)
                result = ToolResult(
                    status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                    output=f"Workflow {workflow_id} deactivated" if success else "",
                    error=None if success else f"Failed to deactivate workflow {workflow_id}",
                )

            elif operation == "history":
                limit = kwargs.get("limit", 10)
                status_filter = kwargs.get("status")
                executions = self.client.get_executions(
                    workflow_id=workflow_id,
                    status=status_filter,
                    limit=limit,
                )

                output_lines = [f"Last {len(executions)} execution(s):"]
                for ex in executions:
                    status_icon = "✓" if ex.get("status") == "success" else "✗"
                    output_lines.append(
                        f"  {status_icon} [{ex.get('id')}] {ex.get('workflowData', {}).get('name', 'Unknown')}"
                    )

                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="\n".join(output_lines),
                    metadata={"executions": executions},
                )

            elif operation == "start":
                success = self.client.start_container()
                result = ToolResult(
                    status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                    output="n8n container started" if success else "",
                    error=None if success else "Failed to start n8n container",
                )

            elif operation == "stop":
                success = self.client.stop_container()
                result = ToolResult(
                    status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                    output="n8n container stopped" if success else "",
                    error=None if success else "Failed to stop n8n container",
                )

            else:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"Unknown operation: {operation}. Valid: health, list, get, run, webhook, activate, deactivate, history, start, stop",
                )

        except Exception as e:
            logger.exception(f"n8n operation failed: {operation}")
            result = ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

        result.execution_time = time.time() - start_time
        self._record_result(result)
        return result


# Global singleton
_n8n_tool: Optional[N8NTool] = None


def get_n8n(
    base_url: str = "http://localhost:5678",
    api_key: Optional[str] = None,
) -> N8NTool:
    """Get the global n8n tool instance."""
    global _n8n_tool
    if _n8n_tool is None:
        _n8n_tool = N8NTool(base_url=base_url, api_key=api_key)
    return _n8n_tool
