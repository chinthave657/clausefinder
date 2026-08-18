# 3GPP TS 38.331 V18.0.0

Some front matter that belongs to no clause.

## Contents

- ignored table of contents lines

## 1 Scope

The present document specifies the Radio Resource Control protocol for the
radio interface between UE and NG-RAN.

## 5 Procedures

## 5.3 RRC connection control

## 5.3.5 RRC reconfiguration

### 5.3.5.3 Reception of an RRCReconfiguration by the UE

The UE shall perform the following actions upon reception of the
RRCReconfiguration, as specified in TS 38.321 clause 5.4.1.
If the UE is unable to comply with any part of the configuration it
shall instead continue with the failure handling specified in
clause 5.3.5.5.

1> if the RRCReconfiguration includes the fullConfig:

2> perform the full configuration procedure as specified in 5.3.5.11;

| Field | Presence | Description |
| ----- | -------- | ----------- |
| fullConfig | Optional | Indicates full configuration |
| masterCellGroup | Optional | Master cell group configuration |

After the table, the UE shall submit the RRCReconfigurationComplete message
to lower layers for transmission.

## Annex A (informative): Guidance

## A.1 Deployment guidance

This annex provides informative deployment guidance for network operators
and is not part of the normative protocol behaviour.
