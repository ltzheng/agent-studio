from datetime import datetime
from typing import Any

from desktop_env.eval.connectors.gspace.gservice import GoogleService


class GoogleCalendarService(GoogleService):
    def __init__(self, token_path: str) -> None:
        super().__init__(
            token_path=token_path,
            service_name="calendar",
            service_version="v3",
        )

    def list_calendars(self) -> list[dict]:
        page_token = None
        calendar_entry_list = []
        while True:
            calendar_list = (
                self.service.calendarList().list(pageToken=page_token).execute()
            )
            for calendar_entry in calendar_list["items"]:
                calendar_entry_list.append(calendar_entry)

            page_token = calendar_list.get("nextPageToken")
            if not page_token:
                break
        return calendar_entry_list

    def create_calendar(self, calendar_info: dict) -> dict[str, str]:
        created_calendar = self.service.calendars().insert(body=calendar_info).execute()
        return created_calendar

    def delete_calendar(self, calendar_id: str) -> bool:
        try:
            self.service.calendars().delete(calendarId=calendar_id).execute()
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def find_calendar_by_id(self, id: str) -> dict[str, str]:
        calendar_entry_list = self.list_calendars()
        for calendar_entry in calendar_entry_list:
            if calendar_entry["id"] == id:
                return calendar_entry
        return {}

    def clear_calendar(self, calendar_id: str) -> None:
        events = self.list_events(calendar_id=calendar_id)

        for event in events:
            self.service.events().delete(
                calendarId=calendar_id, eventId=event["id"]
            ).execute()

    def create_event(
        self,
        start_time: str,
        end_time: str,
        summary: str | None = None,
        location: str | None = None,
        description: str | None = None,
        attendees: list[str] | None = None,
        calendar_id: str = "primary",
        time_zone: str | None = "UTC",
    ) -> dict[str, Any]:
        event_info = {
            "summary": summary,
            "location": location,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": time_zone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": time_zone,
            },
            "attendees": [{"email": attendee} for attendee in attendees]
            if attendees
            else [],
        }
        event = (
            self.service.events()
            .insert(calendarId=calendar_id, body=event_info)
            .execute()
        )
        return event

    def delete_event(self, event_id: str, calendar_id: str = "primary") -> bool:
        try:
            self.service.events().delete(
                calendarId=calendar_id, eventId=event_id
            ).execute()
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def get_event(self, event_id: str, calendar_id: str = "primary") -> dict[str, str]:
        event = (
            self.service.events()
            .get(calendarId=calendar_id, eventId=event_id)
            .execute()
        )
        return event

    def list_events(self, calendar_id: str = "primary") -> list[dict]:
        events = []
        page_token = None
        while True:
            events_result = (
                self.service.events()
                .list(calendarId=calendar_id, pageToken=page_token)
                .execute()
            )
            events.extend(events_result["items"])
            page_token = events_result.get("nextPageToken")
            if not page_token:
                break
        return events

    def update_event(
        self, event_id: str, updated_event: dict, calendar_id: str = "primary"
    ) -> dict[str, str]:
        event = (
            self.service.events()
            .update(calendarId=calendar_id, eventId=event_id, body=updated_event)
            .execute()
        )
        return event

    @staticmethod
    def match_time(
        ref: str,
        pred: str,
    ) -> bool:
        ref_time = datetime.fromisoformat(ref).astimezone().isoformat()
        pred_time = datetime.fromisoformat(pred).astimezone().isoformat()
        return ref_time == pred_time

    def event_match_left(
        self,
        ref: dict,
        pred: dict,
    ) -> bool:
        for key, item in ref.items():
            pred_item = pred.get(key, None)
            if key in ("summary", "description", "location"):
                if item != pred_item:
                    return False
            elif key in ["start", "end"]:
                if not (
                    "dateTime" in item
                    and "dateTime" in pred_item
                    and self.match_time(item["dateTime"], pred_item["dateTime"])
                ):
                    return False
        return True

    def search_events_by_info(
        self, reference_event: dict, calendar_id: str = "primary"
    ) -> list[dict[str, str]]:
        events = self.list_events(calendar_id=calendar_id)
        results = []
        for event in events:
            if self.event_match_left(reference_event, event):
                results.append(event)
        return results

    def search_events_by_time_range(
        self,
        start_time: str,
        end_time: str,
        calendar_id: str = "primary",
    ) -> list[dict[str, Any]]:
        events = []
        page_token = None
        while True:
            events_result = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=start_time,
                    timeMax=end_time,
                    singleEvents=True,
                )
                .execute()
            )
            events.extend(events_result["items"])
            page_token = events_result.get("nextPageToken")
            if not page_token:
                break
        return events
