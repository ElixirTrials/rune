# HITL Flow

```mermaid
sequenceDiagram
    participant Guest as Guest/User
    participant Interface as Communication<br/>Interface
    participant Backend as Backend<br/>Services
    participant ConvAgent as Conversation<br/>Agent
    participant Guard as Guardrail<br/>Agent
    participant Console as Admin<br/>Console
    participant Operator as Human<br/>Operator

    Guest->>Interface: Send Message
    Interface->>Backend: Forward Request
    Backend->>ConvAgent: Process Message
    ConvAgent->>ConvAgent: Analyze Context & History
    ConvAgent->>Guard: Generate Response
    Guard->>Guard: Validate Content

    alt Guardrail Approved
        Guard->>Console: Log Event
        Guard->>Interface: Send Response
        Interface->>Guest: Deliver Message
    else Guardrail Flagged
        Guard->>Console: Alert: Requires Review
        Console->>Operator: Show in Dashboard
        Operator->>Console: Review & Decide
        alt Operator Approves
            Console->>Backend: Approve Response
            Backend->>Interface: Send Response
            Interface->>Guest: Deliver Message
        else Operator Takes Over
            Console->>Backend: Enable Human Mode
            Operator->>Interface: Direct Message
            Interface->>Guest: Deliver Message
        end
    end

    Note over Console: Real-time event stream<br/>updates orchestration panel

    %% Styling
    style Guest fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Interface fill:#e1f5ff,stroke:#007acc,stroke-width:2px
    style Backend fill:#d4f1d4,stroke:#28a745,stroke-width:2px
    style ConvAgent fill:#d4f1d4,stroke:#28a745,stroke-width:2px
    style Guard fill:#ffe5cc,stroke:#fd7e14,stroke-width:2px
    style Console fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    style Operator fill:#f0f0f0,stroke:#666,stroke-width:2px
```
